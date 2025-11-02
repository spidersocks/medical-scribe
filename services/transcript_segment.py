from __future__ import annotations

from __future__ import annotations

import asyncio  # NEW: Import the asyncio library
from typing import List, Optional, Tuple
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS
from datetime import datetime, timezone

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.transcript_segment import (
    TranscriptSegmentCreate,
    TranscriptSegmentRead,
    TranscriptSegmentUpdate,
)
from services import nlp  # translate + comprehend
import time

_CACHE_TTL = 2.0  # seconds
_SEGMENTS_CACHE: dict[tuple[str, bool], tuple[float, list[dict]]] = {}  # (cid, include_entities) -> (ts, serialized list)


def _cache_key(cid: str, include: bool) -> tuple[str, bool]:
    return (cid, include)


def _cache_get(cid: str, include: bool) -> Optional[list[dict]]:
    key = _cache_key(cid, include)
    entry = _SEGMENTS_CACHE.get(key)
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts <= _CACHE_TTL:
        return payload
    _SEGMENTS_CACHE.pop(key, None)
    return None


def _cache_set(cid: str, include: bool, normalized: list[TranscriptSegmentRead]) -> None:
    # Store as plain dicts to avoid retaining pydantic objects
    payload = [item.model_dump() for item in normalized]
    _SEGMENTS_CACHE[_cache_key(cid, include)] = (time.time(), payload)


def _cache_bust(cid: str) -> None:
    _SEGMENTS_CACHE.pop((cid, True), None)
    _SEGMENTS_CACHE.pop((cid, False), None)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(val: Optional[object], default: int = 0) -> int:
    try:
        return int(val)  # type: ignore[arg-type]
    except Exception:
        return default


def _expand_compact(compact) -> List[dict]:
    """Expand compact entity spans into the UI-compatible shape."""
    if not isinstance(compact, list):
        return []
    out: List[dict] = []
    for e in compact:
        try:
            out.append(
                {
                    "BeginOffset": int(e.get("b", 0)),
                    "EndOffset": int(e.get("e", 0)),
                    "Category": str(e.get("c", "OTHER")),
                    "Type": str(e.get("y", "OTHER")),
                }
            )
        except Exception:
            continue
    return out


def _compact_from_entities(entities_list: List[dict]) -> List[dict]:
    compact: List[dict] = []
    for e in entities_list or []:
        try:
            compact.append(
                {
                    "b": int(e.get("BeginOffset", 0)),
                    "e": int(e.get("EndOffset", 0)),
                    "c": str(e.get("Category", "OTHER")),
                    "y": str(e.get("Type", "OTHER")),
                }
            )
        except Exception:
            continue
    return compact



class TranscriptSegmentService(DynamoServiceMixin):
    async def _query_segments_for_consultation(self, consultation_id: str) -> List[dict]:
        # unchanged
        items: List[dict] = []
        kwargs = {"KeyConditionExpression": Key("consultationId").eq(consultation_id)}
        response = await run_in_thread(self.table.query, **kwargs)
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = await run_in_thread(self.table.query, **kwargs)
            items.extend(response.get("Items", []))
        return [self.clean(it) for it in items]

    async def list_for_consultation(
        self,
        consultation_id: str | UUID,
        *,
        include_entities: bool = False,
    ) -> List[TranscriptSegmentRead]:
        cid = str(consultation_id)

        # Fast path: serve from short-lived memory cache if available
        cached = _cache_get(cid, include_entities)
        if cached is not None:
            return [TranscriptSegmentRead.model_validate(it) for it in cached]

        items = await self._query_segments_for_consultation(cid)
        normalized: List[TranscriptSegmentRead] = []

        for it in items:
            # ... normalization logic unchanged ...
            consultation_id_norm = str(it.get("consultation_id") or it.get("consultationId") or cid)
            sequence_number = _to_int(it.get("sequence_number") or it.get("segmentIndex"), 0)
            speaker_label = it.get("speaker_label") or it.get("speaker") or None
            original_text = it.get("original_text") or it.get("text") or it.get("displayText") or ""
            translated_text = it.get("translated_text") or it.get("translatedText") or None
            detected_language = it.get("detected_language") or None
            created_at = it.get("created_at") or _now_iso()

            entities_list = _expand_compact(it.get("entities_compact") or [])
            translated_override: Optional[str] = None
            entities_ref = it.get("entities_ref")

            if include_entities and not entities_list:
                if detected_language and not str(detected_language).lower().startswith("en"):
                    text_en = nlp.to_english(original_text, detected_language)
                    translated_override = text_en if text_en and text_en != translated_text else None
                    analysis_text = text_en
                    entities_ref = "translated"
                else:
                    analysis_text = original_text
                    entities_ref = "original"

                try:
                    entities_list = nlp.detect_entities(analysis_text)
                except Exception:
                    entities_list = []

                compact = _compact_from_entities(entities_list)
                try:
                    merged = dict(it)
                    merged["entities_compact"] = compact
                    merged["entities_ref"] = entities_ref
                    merged["entities"] = None
                    if translated_override:
                        merged["translated_text"] = translated_override
                        merged["translatedText"] = translated_override
                    merged["consultationId"] = merged.get("consultationId") or consultation_id_norm
                    merged["segmentIndex"] = merged.get("segmentIndex") or sequence_number
                    await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
                    it = self.clean(merged)
                except Exception:
                    pass
            else:
                entities_ref = entities_ref or (
                    "translated" if translated_text and not (detected_language or "").lower().startswith("en") else "original"
                )

            seg_id = it.get("segment_id") or str(uuid5(NAMESPACE_DNS, f"{consultation_id_norm}:{sequence_number}"))

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
                "entities": entities_list,
                "entities_compact": it.get("entities_compact") or None,
                "entities_ref": entities_ref,
                "created_at": created_at,
            }
            normalized.append(TranscriptSegmentRead.model_validate(data))

        normalized.sort(key=lambda x: x.sequence_number)
        # Populate cache
        _cache_set(cid, include_entities, normalized)
        return normalized

    async def enrich_consultation(
        self,
        consultation_id: str | UUID,
        *,
        force: bool = False,
        concurrency: int = 5,
    ) -> dict:
        """
        Precompute and cache entities_compact (and translated_text if needed) for all segments.
        Returns counts: total, enriched, skipped.
        """
        cid = str(consultation_id)
        items = await self._query_segments_for_consultation(cid)

        to_enrich = []
        for it in items:
            has_compact = bool(it.get("entities_compact"))
            if force or not has_compact:
                to_enrich.append(it)

        sem = asyncio.Semaphore(max(1, concurrency))
        enriched = 0

        async def _enrich_one(it: dict) -> None:
            nonlocal enriched
            async with sem:
                sequence_number = _to_int(it.get("sequence_number") or it.get("segmentIndex"), 0)
                original_text = it.get("original_text") or it.get("text") or it.get("displayText") or ""
                detected_language = it.get("detected_language")
                translated_text = it.get("translated_text") or it.get("translatedText")
                consultation_id_norm = str(it.get("consultation_id") or it.get("consultationId") or cid)

                # Determine analysis text & ref
                if detected_language and not str(detected_language).lower().startswith("en"):
                    text_en = await asyncio.to_thread(nlp.to_english, original_text, detected_language)
                    translated_override = text_en if text_en and text_en != translated_text else None
                    analysis_text = text_en
                    entities_ref = "translated"
                else:
                    translated_override = None
                    analysis_text = original_text
                    entities_ref = "original"

                try:
                    entities_list = await asyncio.to_thread(nlp.detect_entities, analysis_text)
                except Exception:
                    entities_list = []

                compact = _compact_from_entities(entities_list)

                try:
                    merged = dict(it)
                    merged["entities_compact"] = compact
                    merged["entities_ref"] = entities_ref
                    merged["entities"] = None
                    if translated_override:
                        merged["translated_text"] = translated_override
                        merged["translatedText"] = translated_override
                    merged["consultationId"] = merged.get("consultationId") or consultation_id_norm
                    merged["segmentIndex"] = merged.get("segmentIndex") or sequence_number
                    await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
                    enriched += 1
                except Exception:
                    # ignore single-segment failures
                    pass

        await asyncio.gather(*[_enrich_one(it) for it in to_enrich])

        return {
            "consultation_id": cid,
            "total": len(items),
            "enriched": enriched,
            "skipped": len(items) - enriched,
            "force": force,
        }

    async def create(self, payload: TranscriptSegmentCreate) -> TranscriptSegmentRead:
        # unchanged except cache bust
        data = payload.model_dump()
        consultation_id = str(data.get("consultation_id"))
        if not consultation_id or consultation_id == "None":
            raise ValueError("consultation_id is required at service layer")

        # bust cache for this consultation
        _cache_bust(consultation_id)

        segment_index = _to_int(data.get("sequence_number"), 0)
        segment_id = str(uuid4())
        created_at = _now_iso()
        item = {
            "consultationId": consultation_id,
            "segmentIndex": segment_index,
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
            "entities": None,
            "entities_compact": None,
            "entities_ref": None,
            "created_at": created_at,
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
            "entities": [],
            "entities_compact": None,
            "entities_ref": None,
            "created_at": created_at,
        }
        return TranscriptSegmentRead.model_validate(response_data)

    async def get(self, segment_id: str | UUID) -> TranscriptSegmentRead:
        items = await self.scan_all()
        for it in items:
            if str(it.get("segment_id")) == str(segment_id):
                entities_list = _expand_compact(it.get("entities_compact") or [])
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
                    "entities_compact": it.get("entities_compact") or None,
                    "entities_ref": it.get("entities_ref") or None,
                    "created_at": it.get("created_at") or _now_iso(),
                }
                return TranscriptSegmentRead.model_validate(data)
        raise NotFoundError(f"Transcript segment with id={segment_id} was not found.")

    async def update(self, segment_id: str | UUID, payload: TranscriptSegmentUpdate) -> TranscriptSegmentRead:
        # unchanged except cache bust
        items = await self.scan_all()
        target = next((it for it in items if str(it.get("segment_id")) == str(segment_id)), None)
        if not target:
            raise NotFoundError(f"Transcript segment with id={segment_id} was not found.")

        # bust cache for this consultation
        cid = str(target.get("consultation_id") or target.get("consultationId") or "")
        if cid:
            _cache_bust(cid)

        updates = payload.model_dump(exclude_unset=True)
        merged = {**target, **self.serialize_input(updates)}
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))

        entities_list = _expand_compact(merged.get("entities_compact") or [])
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
            "entities_compact": merged.get("entities_compact") or None,
            "entities_ref": merged.get("entities_ref") or None,
            "created_at": merged.get("created_at") or _now_iso(),
        }
        return TranscriptSegmentRead.model_validate(data)

    async def delete(self, segment_id: str | UUID) -> None:
        items = await self.scan_all()
        target = next((it for it in items if str(it.get("segment_id")) == str(segment_id)), None)
        if not target:
            raise NotFoundError(f"Transcript segment with id={segment_id} was not found.")

        cid = str(target.get("consultationId") or target.get("consultation_id") or "")
        if cid:
            _cache_bust(cid)

        key = {"consultationId": target["consultationId"], "segmentIndex": target["segmentIndex"]}
        await run_in_thread(self.table.delete_item, Key=key)

transcript_segment_service = TranscriptSegmentService(
    table_env_name="TRANSCRIPT_SEGMENTS_TABLE_NAME",
    default_table_name="medical-scribe-transcript-segments",
    partition_key="consultationId",
)