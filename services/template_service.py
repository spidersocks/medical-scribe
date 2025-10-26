from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.template import TemplateCreate, TemplateRead, TemplateUpdate


class TemplateService(DynamoServiceMixin):
    async def list_for_user(self, owner_user_id: str) -> List[TemplateRead]:
        # For small scale we can scan then filter by owner_user_id (keeping consistent with other services)
        items = await self.scan_all()
        filtered = [item for item in items if str(item.get("ownerUserId") or item.get("owner_user_id") or "") == owner_user_id]
        return [TemplateRead.model_validate(item) for item in filtered]

    async def create(self, owner_user_id: str, payload: TemplateCreate) -> TemplateRead:
        data = payload.model_dump()
        template_id = str(uuid4())
        now = datetime.utcnow().isoformat()
        item = {
            "id": template_id,
            "ownerUserId": owner_user_id,
            "name": data["name"],
            "sections": data["sections"],
            "example_text": data.get("example_text") or "",
            "created_at": now,
            "updated_at": now,
        }
        await run_in_thread(self.table.put_item, Item=self.serialize_input(item))
        return TemplateRead.model_validate(self.clean(item))

    async def get(self, template_id: str) -> TemplateRead:
        item = await self.get_required_item(str(template_id))
        return TemplateRead.model_validate(item)

    async def update(self, template_id: str, payload: TemplateUpdate) -> TemplateRead:
        existing = await self.get_required_item(str(template_id))
        updates = payload.model_dump(exclude_unset=True)
        merged = {**existing, **self.serialize_input(updates)}
        merged["updated_at"] = datetime.utcnow().isoformat()
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
        return TemplateRead.model_validate(self.clean(merged))

    async def delete(self, template_id: str) -> None:
        try:
            await run_in_thread(
                self.table.delete_item,
                Key={self.partition_key: str(template_id)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": self.partition_key},
            )
        except ClientError as exc:
            # Mirror other services' NotFound handling
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"Template with id={template_id} was not found.") from exc
            raise


template_service = TemplateService(
    table_env_name="TEMPLATES_TABLE_NAME",
    default_table_name="medical-scribe-templates",
)