from __future__ import annotations

from typing import Optional
from uuid import UUID, uuid4

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError, ValidationError
from schemas.clinical_note import (
    ClinicalNoteCreate,
    ClinicalNoteRead,
    ClinicalNoteUpdate,
)


class ClinicalNoteService(DynamoServiceMixin):
    async def create(self, payload: ClinicalNoteCreate) -> ClinicalNoteRead:
        data = payload.model_dump()
        consultation_id = data.get("consultation_id")
        if not consultation_id:
            raise ValidationError("consultation_id is required to create a clinical note.")
        consultation_id_str = str(consultation_id)

        existing = await self.try_get_by_consultation(consultation_id_str)
        if existing:
            raise ValidationError("A clinical note already exists for this consultation.")

        raw_id = data.pop("id", None)
        note_id = str(raw_id or uuid4())
        item = {"id": note_id, "consultation_id": consultation_id_str, **self.serialize_input(data)}
        await run_in_thread(self.table.put_item, Item=item)
        clean_item = self.clean(item)
        return ClinicalNoteRead.model_validate(clean_item)

    async def get(self, note_id: str | UUID) -> ClinicalNoteRead:
        item = await self.get_required_item(str(note_id))
        return ClinicalNoteRead.model_validate(item)

    async def get_by_consultation(self, consultation_id: str | UUID) -> ClinicalNoteRead:
        note = await self.try_get_by_consultation(str(consultation_id))
        if not note:
            raise NotFoundError(f"No clinical note found for consultation {consultation_id}.")
        return note

    async def try_get_by_consultation(
        self,
        consultation_id: str | UUID,
    ) -> Optional[ClinicalNoteRead]:
        consultation_id_str = str(consultation_id)
        items = await self.scan_all()
        for item in items:
            if str(item.get("consultation_id")) == consultation_id_str:
                return ClinicalNoteRead.model_validate(item)
        return None

    async def update(
        self,
        note_id: str | UUID,
        payload: ClinicalNoteUpdate,
    ) -> ClinicalNoteRead:
        existing = await self.get_required_item(str(note_id))
        updates = payload.model_dump(exclude_unset=True)
        merged = {**existing, **self.serialize_input(updates)}
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
        clean_item = self.clean(merged)
        return ClinicalNoteRead.model_validate(clean_item)

    async def upsert(
        self,
        consultation_id: str | UUID,
        payload: ClinicalNoteUpdate,
    ) -> ClinicalNoteRead:
        existing = await self.try_get_by_consultation(str(consultation_id))
        if existing:
            return await self.update(existing.id, payload)  # type: ignore[attr-defined]

        create_payload = ClinicalNoteCreate.model_validate(
            {"consultation_id": str(consultation_id), **payload.model_dump()}
        )
        return await self.create(create_payload)

    async def delete(self, note_id: str | UUID) -> None:
        try:
            await run_in_thread(
                self.table.delete_item,
                Key={self.partition_key: str(note_id)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": self.partition_key},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"Clinical note with id={note_id} was not found.") from exc
            raise


clinical_note_service = ClinicalNoteService(
    table_env_name="CLINICAL_NOTES_TABLE_NAME",
    default_table_name="medical-scribe-clinical-notes",
)