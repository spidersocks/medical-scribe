from __future__ import annotations

from typing import List, Optional
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS
from datetime import datetime, timezone

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.transcript_segment import (
    TranscriptSegmentCreate,
    TranscriptSegmentRead,
    TranscriptSegmentUpdate,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(val: Optional[object], default: int = 0) -> int:
    try:
        return int(val)  # type: ignore[arg-type]
    except Exception:
        return default


class TranscriptSegmentService(DynamoServiceMixin):
    async def list_for_consultation(
        self,
        consultation_id: str | UUID,
    ) -> List[TranscriptSegmentRead]:
        """
        Read-all via scan (kept simple), normalize legacy (camelCase) and new (snake_case)
        shapes into TranscriptSegmentRead.
        """
        cid = str(consultation_id)
        items = await self.scan_all()

        # Filter by either consultationId (legacy) or consultation_id (new)
        filtered = [
            item
            for item in items
            if str(item.get("consultation_id") or item.get("consultationId") or "") == cid
        ]

        # Normalize each item to the read-model fields
        normalized = []
        for it in filtered:
            # Prefer snake_case, fall back to legacy camelCase
            consultation_id_norm = str(it.get("consultation_id") or it.get("consultationId") or cid)
            sequence_number = _to_int(it.get("sequence_number") or it.get("segmentIndex"), 0)
            speaker_label = it.get("speaker_label") or it.get("speaker") or None
            original_text = (
                it.get("original_text")
                or it.get("text")
                or it.get("displayText")
                or ""
            )
            translated_text = it.get("translated_text") or it.get("translatedText") or None
            detected_language = it.get("detected_language") or None
            created_at = it.get("created_at") or _now_iso()

            # Ensure a stable segment_id:
            seg_id = it.get("segment_id")
            if not seg_id:
                # Deterministic UUID derived from (consultation_id, sequence_number)
                seg_id = str(uuid5(NAMESPACE_DNS, f"{consultation_id_norm}:{sequence_number}"))

            data = {
                "segment_id": seg_id,
                "consultation_id": consultation_id_norm,
                "sequence_number": sequence_number,
                "speaker_label": speaker_label,
                "speaker_role": it.get("speaker_role") or None,
                "original_text": original_text,
                "translated_text": translated_text,
                "detected_language": detected_language,
                "start_time_ms": it.get("start_time_ms"),
                "end_time_ms": it.get("end_time_ms"),
                "entities": it.get("entities"),
                "created_at": created_at,
            }
            normalized.append(TranscriptSegmentRead.model_validate(data))

        # Sort by sequence_number for stable reconstruction
        normalized.sort(key=lambda x: x.sequence_number)
        return normalized

    async def create(
        self,
        payload: TranscriptSegmentCreate,
    ) -> TranscriptSegmentRead:
        """
        Write into the existing table schema that uses:
          - PK: consultationId (string)
          - SK: segmentIndex (number)
        Also store snake_case fields so API read-model maps cleanly.
        """
        data = payload.model_dump()
        # Route injected this; validation now allows body without it
        consultation_id = str(data.get("consultation_id"))
        if not consultation_id or consultation_id == "None":
            raise ValueError("consultation_id is required at service layer")

        # Required for legacy table key
        segment_index = _to_int(data.get("sequence_number"), 0)

        # Generate server-side fields
        segment_id = str(uuid4())
        created_at = _now_iso()

        # Compose item merging both worlds:
        # - legacy keys for table primary key
        # - snake_case fields for clean API reading
        item = {
            # Legacy table keys
            "consultationId": consultation_id,
            "segmentIndex": segment_index,

            # Keep new/clean schema as first-class fields
            "segment_id": segment_id,
            "consultation_id": consultation_id,
            "sequence_number": segment_index,  # mirror for clarity
            "speaker_label": data.get("speaker_label"),
            "speaker_role": data.get("speaker_role"),
            "original_text": data.get("original_text") or "",
            "translated_text": data.get("translated_text"),
            "detected_language": data.get("detected_language"),
            "start_time_ms": data.get("start_time_ms"),
            "end_time_ms": data.get("end_time_ms"),
            "entities": data.get("entities"),
            "created_at": created_at,

            # Optional legacy convenience mirrors (not required but helps if you ever fall back)
            "text": data.get("original_text") or "",
            "displayText": data.get("original_text") or "",
            "translatedText": data.get("translated_text"),
        }

        await run_in_thread(self.table.put_item, Item=self.serialize_input(item))

        # IMPORTANT: Return only the API read model fields (no legacy/camelCase extras)
        response_data = {
            "segment_id": segment_id,
            "consultation_id": consultation_id,
            "sequence_number": segment_index,
            "speaker_label": item.get("speaker_label"),
            "speaker_role": item.get("speaker_role"),
            "original_text": item.get("original_text") or "",
            "translated_text": item.get("translated_text"),
            "detected_language": item.get("detected_language"),
            "start_time_ms": item.get("start_time_ms"),
            "end_time_ms": item.get("end_time_ms"),
            "entities": item.get("entities"),
            "created_at": created_at,
        }
        return TranscriptSegmentRead.model_validate(response_data)

    async def get(self, segment_id: str | UUID) -> TranscriptSegmentRead:
        # Not used by the UI today; fallback: scan then match by segment_id
        items = await self.scan_all()
        for it in items:
            if str(it.get("segment_id")) == str(segment_id):
                # Normalize to read model
                data = {
                    "segment_id": str(it.get("segment_id")),
                    "consultation_id": str(it.get("consultation_id") or it.get("consultationId") or ""),
                    "sequence_number": _to_int(it.get("sequence_number") or it.get("segmentIndex"), 0),
                    "speaker_label": it.get("speaker_label") or it.get("speaker") or None,
                    "speaker_role": it.get("speaker_role") or None,
                    "original_text": it.get("original_text") or it.get("text") or "",
                    "translated_text": it.get("translated_text") or it.get("translatedText") or None,
                    "detected_language": it.get("detected_language") or None,
                    "start_time_ms": it.get("start_time_ms"),
                    "end_time_ms": it.get("end_time_ms"),
                    "entities": it.get("entities"),
                    "created_at": it.get("created_at") or _now_iso(),
                }
                return TranscriptSegmentRead.model_validate(data)
        raise NotFoundError(f"Transcript segment with id={segment_id} was not found.")

    async def update(
        self,
        segment_id: str | UUID,
        payload: TranscriptSegmentUpdate,
    ) -> TranscriptSegmentRead:
        # Fallback: scan to find the full item (to get its legacy keys), then replace
        items = await self.scan_all()
        target = None
        for it in items:
            if str(it.get("segment_id")) == str(segment_id):
                target = it
                break
        if not target:
            raise NotFoundError(f"Transcript segment with id={segment_id} was not found.")

        updates = payload.model_dump(exclude_unset=True)
        merged = {**target, **self.serialize_input(updates)}
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))

        # Normalize to read model for response
        data = {
            "segment_id": str(merged.get("segment_id")),
            "consultation_id": str(merged.get("consultation_id") or merged.get("consultationId") or ""),
            "sequence_number": _to_int(merged.get("sequence_number") or merged.get("segmentIndex"), 0),
            "speaker_label": merged.get("speaker_label") or merged.get("speaker") or None,
            "speaker_role": merged.get("speaker_role") or None,
            "original_text": merged.get("original_text") or merged.get("text") or "",
            "translated_text": merged.get("translated_text") or merged.get("translatedText") or None,
            "detected_language": merged.get("detected_language") or None,
            "start_time_ms": merged.get("start_time_ms"),
            "end_time_ms": merged.get("end_time_ms"),
            "entities": merged.get("entities"),
            "created_at": merged.get("created_at") or _now_iso(),
        }
        return TranscriptSegmentRead.model_validate(data)

    async def delete(self, segment_id: str | UUID) -> None:
        # Fallback: scan to discover legacy keys and then delete by those keys
        items = await self.scan_all()
        target = None
        for it in items:
            if str(it.get("segment_id")) == str(segment_id):
                target = it
                break
        if not target:
            raise NotFoundError(f"Transcript segment with id={segment_id} was not found.")

        key = {
            "consultationId": target["consultationId"],
            "segmentIndex": target["segmentIndex"],
        }
        try:
            await run_in_thread(
                self.table.delete_item,
                Key=key,
                ConditionExpression="attribute_exists(consultationId) AND attribute_exists(segmentIndex)",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"Transcript segment with id={segment_id} was not found.") from exc
            raise


# IMPORTANT: keep the existing table name, but do not set partition_key to segment_id anymore,
# because the real table uses a composite key (consultationId + segmentIndex).
# We won't use DynamoServiceMixin.get_required_item (which needs a single key).
transcript_segment_service = TranscriptSegmentService(
    table_env_name="TRANSCRIPT_SEGMENTS_TABLE_NAME",
    default_table_name="medical-scribe-transcript-segments",
    partition_key="consultationId",  # only used by generic helpers; we avoid them for get/delete
)