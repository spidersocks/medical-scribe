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
from services import nlp  # NEW

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
        *,
        include_entities: bool = False,  # NEW
    ) -> List[TranscriptSegmentRead]:
        """
        Normalize legacy/new rows and optionally enrich with Comprehend Medical.
        Always return entities as a list for frontend highlighting.
        """
        cid = str(consultation_id)
        items = await self.scan_all()

        # Filter rows for this consultation
        filtered = [
            item
            for item in items
            if str(item.get("consultation_id") or item.get("consultationId") or "") == cid
        ]

        normalized: List[TranscriptSegmentRead] = []
        for it in filtered:
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

            # Normalize entities to a list of dicts:
            entities_val = it.get("entities")
            if isinstance(entities_val, dict) and "Entities" in entities_val:
                entities_list = entities_val.get("Entities") or []
            elif isinstance(entities_val, list):
                entities_list = entities_val
            else:
                entities_list = []

            translated_override = None
            if include_entities:
                text_en = nlp.to_english(original_text, detected_language)
                try:
                    entities_list = nlp.detect_entities(text_en)
                except Exception:
                    pass
                # If we translated (non-English source), surface translated_text so UI can highlight it
                if detected_language and not str(detected_language).lower().startswith("en"):
                    if text_en and text_en != original_text:
                        translated_override = text_en

            # Ensure a stable segment_id for legacy rows
            seg_id = it.get("segment_id")
            if not seg_id:
                seg_id = str(uuid5(NAMESPACE_DNS, f"{consultation_id_norm}:{sequence_number}"))

            data = {
                "segment_id": seg_id,
                "consultation_id": consultation_id_norm,
                "sequence_number": sequence_number,
                "speaker_label": speaker_label,
                "speaker_role": it.get("speaker_role") or None,
                "original_text": original_text,
                "translated_text": translated_override if translated_override is not None else translated_text,
                "detected_language": detected_language,
                "start_time_ms": it.get("start_time_ms"),
                "end_time_ms": it.get("end_time_ms"),
                "entities": entities_list,  # ALWAYS a list on read
                "created_at": created_at,
            }
            normalized.append(TranscriptSegmentRead.model_validate(data))

    async def create(
        self,
        payload: TranscriptSegmentCreate,
    ) -> TranscriptSegmentRead:
        """
        Write into the existing table schema (consultationId + segmentIndex).
        Store snake_case fields for clean reads. We do not persist entities here.
        """
        data = payload.model_dump()
        consultation_id = str(data.get("consultation_id"))
        if not consultation_id or consultation_id == "None":
            raise ValueError("consultation_id is required at service layer")

        segment_index = _to_int(data.get("sequence_number"), 0)
        segment_id = str(uuid4())
        created_at = _now_iso()

        item = {
            # Legacy table keys
            "consultationId": consultation_id,
            "segmentIndex": segment_index,

            # Clean schema fields (snake_case)
            "segment_id": segment_id,
            "consultation_id": consultation_id,
            "sequence_number": segment_index,
            "speaker_label": data.get("speaker_label"),
            "speaker_role": data.get("speaker_role"),
            "original_text": data.get("original_text") or "",
            "translated_text": data.get("translated_text"),
            "detected_language": data.get("detected_language"),
            "start_time_ms": data.get("start_time_ms"),
            "end_time_ms": data.get("end_time_ms"),
            "entities": None,  # not persisted
            "created_at": created_at,

            # Optional legacy mirrors
            "text": data.get("original_text") or "",
            "displayText": data.get("original_text") or "",
            "translatedText": data.get("translated_text"),
        }

        await run_in_thread(self.table.put_item, Item=self.serialize_input(item))

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
            "entities": [],  # return empty list to UI
            "created_at": created_at,
        }
        return TranscriptSegmentRead.model_validate(response_data)

    async def get(self, segment_id: str | UUID) -> TranscriptSegmentRead:
        items = await self.scan_all()
        for it in items:
            if str(it.get("segment_id")) == str(segment_id):
                # Normalize entities to a list on read
                entities_val = it.get("entities")
                if isinstance(entities_val, dict) and "Entities" in entities_val:
                    entities_list = entities_val.get("Entities") or []
                elif isinstance(entities_val, list):
                    entities_list = entities_val
                else:
                    entities_list = []

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
                    "entities": entities_list,
                    "created_at": it.get("created_at") or _now_iso(),
                }
                return TranscriptSegmentRead.model_validate(data)
        raise NotFoundError(f"Transcript segment with id={segment_id} was not found.")

    async def update(
        self,
        segment_id: str | UUID,
        payload: TranscriptSegmentUpdate,
    ) -> TranscriptSegmentRead:
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

        # Normalize entities to list
        entities_val = merged.get("entities")
        if isinstance(entities_val, dict) and "Entities" in entities_val:
            entities_list = entities_val.get("Entities") or []
        elif isinstance(entities_val, list):
            entities_list = entities_val
        else:
            entities_list = []

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
            "entities": entities_list,
            "created_at": merged.get("created_at") or _now_iso(),
        }
        return TranscriptSegmentRead.model_validate(data)

    async def delete(self, segment_id: str | UUID) -> None:
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
        await run_in_thread(self.table.delete_item, Key=key)


transcript_segment_service = TranscriptSegmentService(
    table_env_name="TRANSCRIPT_SEGMENTS_TABLE_NAME",
    default_table_name="medical-scribe-transcript-segments",
    partition_key="consultationId",
)