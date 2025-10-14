from __future__ import annotations

from typing import List
from uuid import UUID, uuid4

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.patient import (
    PatientCreate,
    PatientRead,
    PatientSummary,
    PatientUpdate,
)


class PatientService(DynamoServiceMixin):
    async def create(self, payload: PatientCreate) -> PatientRead:
        data = payload.model_dump()
        raw_id = data.pop("id", None)
        patient_id = str(raw_id or uuid4())
        item = {"id": patient_id, **self.serialize_input(data)}
        await run_in_thread(self.table.put_item, Item=item)
        clean_item = self.clean(item)
        return PatientRead.model_validate(clean_item)

    async def list_for_user(
        self,
        user_id: str | UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        starred_only: bool = False,
        summary: bool = False,
    ) -> List[PatientSummary] | List[PatientRead]:
        user_id_str = str(user_id)
        items = await self.scan_all()
        filtered = [
            item
            for item in items
            if str(item.get("user_id")) == user_id_str
            and (not starred_only or bool(item.get("starred")))
        ]
        paginated = filtered[offset : offset + limit]
        if summary:
            return [PatientSummary.model_validate(item) for item in paginated]
        return [PatientRead.model_validate(item) for item in paginated]

    async def get(self, patient_id: str | UUID) -> PatientRead:
        item = await self.get_required_item(str(patient_id))
        return PatientRead.model_validate(item)

    async def update(self, patient_id: str | UUID, payload: PatientUpdate) -> PatientRead:
        existing = await self.get_required_item(str(patient_id))
        updates = payload.model_dump(exclude_unset=True)
        merged = {**existing, **self.serialize_input(updates)}
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
        clean_item = self.clean(merged)
        return PatientRead.model_validate(clean_item)

    async def delete(self, patient_id: str | UUID) -> None:
        try:
            await run_in_thread(
                self.table.delete_item,
                Key={self.partition_key: str(patient_id)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": self.partition_key},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"Patient with id={patient_id} was not found.") from exc
            raise


patient_service = PatientService(
    table_env_name="PATIENTS_TABLE_NAME",
    default_table_name="medical-scribe-patients",
)