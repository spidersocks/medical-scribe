from __future__ import annotations

from typing import List, Optional
from uuid import UUID, uuid4

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.consultation import (
    ConsultationCreate,
    ConsultationRead,
    ConsultationSummary,
    ConsultationUpdate,
)


class ConsultationService(DynamoServiceMixin):
    async def create(self, payload: ConsultationCreate) -> ConsultationRead:
        data = payload.model_dump()
        raw_id = data.pop("id", None)
        consultation_id = str(raw_id or uuid4())
        item = {"id": consultation_id, **self.serialize_input(data)}
        await run_in_thread(self.table.put_item, Item=item)
        clean_item = self.clean(item)
        return ConsultationRead.model_validate(clean_item)

    async def list_for_user(
        self,
        user_id: str | UUID,
        *,
        patient_id: Optional[str | UUID] = None,
        limit: int = 100,
        offset: int = 0,
        summary: bool = True,
    ) -> List[ConsultationSummary] | List[ConsultationRead]:
        user_id_str = str(user_id)
        patient_id_str = str(patient_id) if patient_id else None
        items = await self.scan_all()
        filtered = [
            item
            for item in items
            if str(item.get("user_id")) == user_id_str
            and (patient_id_str is None or str(item.get("patient_id")) == patient_id_str)
        ]
        paginated = filtered[offset : offset + limit]
        if summary:
            return [ConsultationSummary.model_validate(item) for item in paginated]
        return [ConsultationRead.model_validate(item) for item in paginated]

    async def list_for_patient(
        self,
        patient_id: str | UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        summary: bool = False,
    ) -> List[ConsultationSummary] | List[ConsultationRead]:
        patient_id_str = str(patient_id)
        items = await self.scan_all()
        filtered = [
            item
            for item in items
            if str(item.get("patient_id")) == patient_id_str
        ]
        paginated = filtered[offset : offset + limit]
        if summary:
            return [ConsultationSummary.model_validate(item) for item in paginated]
        return [ConsultationRead.model_validate(item) for item in paginated]

    async def get(self, consultation_id: str | UUID) -> ConsultationRead:
        item = await self.get_required_item(str(consultation_id))
        return ConsultationRead.model_validate(item)

    async def update(
        self,
        consultation_id: str | UUID,
        payload: ConsultationUpdate,
    ) -> ConsultationRead:
        existing = await self.get_required_item(str(consultation_id))
        updates = payload.model_dump(exclude_unset=True)
        merged = {**existing, **self.serialize_input(updates)}
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
        clean_item = self.clean(merged)
        return ConsultationRead.model_validate(clean_item)

    async def delete(self, consultation_id: str | UUID) -> None:
        try:
            await run_in_thread(
                self.table.delete_item,
                Key={self.partition_key: str(consultation_id)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": self.partition_key},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"Consultation with id={consultation_id} was not found.") from exc
            raise


consultation_service = ConsultationService(
    table_env_name="CONSULTATIONS_TABLE_NAME",
    default_table_name="medical-scribe-consultations",
)