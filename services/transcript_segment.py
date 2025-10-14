from __future__ import annotations

from typing import List
from uuid import UUID, uuid4

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.transcript_segment import (
    TranscriptSegmentCreate,
    TranscriptSegmentRead,
    TranscriptSegmentUpdate,
)


class TranscriptSegmentService(DynamoServiceMixin):
    async def list_for_consultation(
        self,
        consultation_id: str | UUID,
    ) -> List[TranscriptSegmentRead]:
        consultation_id_str = str(consultation_id)
        items = await self.scan_all()
        filtered = [
            item
            for item in items
            if str(item.get("consultation_id")) == consultation_id_str
        ]
        return [TranscriptSegmentRead.model_validate(item) for item in filtered]

    async def create(
        self,
        payload: TranscriptSegmentCreate,
    ) -> TranscriptSegmentRead:
        data = payload.model_dump()
        raw_id = data.pop("id", None)
        segment_id = str(raw_id or uuid4())
        item = {"id": segment_id, **self.serialize_input(data)}
        await run_in_thread(self.table.put_item, Item=item)
        clean_item = self.clean(item)
        return TranscriptSegmentRead.model_validate(clean_item)

    async def get(self, segment_id: str | UUID) -> TranscriptSegmentRead:
        item = await self.get_required_item(str(segment_id))
        return TranscriptSegmentRead.model_validate(item)

    async def update(
        self,
        segment_id: str | UUID,
        payload: TranscriptSegmentUpdate,
    ) -> TranscriptSegmentRead:
        existing = await self.get_required_item(str(segment_id))
        updates = payload.model_dump(exclude_unset=True)
        merged = {**existing, **self.serialize_input(updates)}
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
        clean_item = self.clean(merged)
        return TranscriptSegmentRead.model_validate(clean_item)

    async def delete(self, segment_id: str | UUID) -> None:
        try:
            await run_in_thread(
                self.table.delete_item,
                Key={self.partition_key: str(segment_id)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": self.partition_key},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"Transcript segment with id={segment_id} was not found.") from exc
            raise


transcript_segment_service = TranscriptSegmentService(
    table_env_name="TRANSCRIPT_SEGMENTS_TABLE_NAME",
    default_table_name="medical-scribe-transcript-segments",
)