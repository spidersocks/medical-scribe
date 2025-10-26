from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import uuid4
import json

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.template import TemplateCreate, TemplateRead, TemplateUpdate


class TemplateService(DynamoServiceMixin):
    async def _normalize_item_for_read(self, raw: dict) -> dict:
        """
        Normalize a raw Dynamo item (cleaned) into the shape expected by TemplateRead.
        - Parse sections if stored as JSON string.
        - Map camelCase createdAt/updatedAt -> created_at/updated_at.
        - Map exampleNoteText -> example_text.
        - Remove any leftover camelCase keys so Pydantic with extra="forbid" won't fail.
        """
        it = dict(raw)  # shallow copy

        # Normalize sections: stringified JSON -> list
        sections = it.get("sections")
        if isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                it["sections"] = parsed if isinstance(parsed, list) else []
            except Exception:
                it["sections"] = []
        elif sections is None:
            it["sections"] = []

        # Normalize example text key variants
        if "exampleNoteText" in it and "example_text" not in it:
            it["example_text"] = it.pop("exampleNoteText")
        elif "exampleNoteText" in it and "example_text" in it:
            # prefer snake_case, remove camelCase
            del it["exampleNoteText"]

        # Normalize timestamps (camelCase -> snake_case).
        # Always remove camelCase keys afterwards to avoid extra-forbid failures.
        if "createdAt" in it:
            if "created_at" not in it:
                it["created_at"] = it.pop("createdAt")
            else:
                # snake exists, remove camel
                del it["createdAt"]

        if "updatedAt" in it:
            if "updated_at" not in it:
                it["updated_at"] = it.pop("updatedAt")
            else:
                del it["updatedAt"]

        # Defensive: if still missing created_at/updated_at, try other keys
        if "created_at" not in it and "createdAt" in raw:
            it["created_at"] = raw.get("createdAt")
        if "updated_at" not in it and "updatedAt" in raw:
            it["updated_at"] = raw.get("updatedAt")

        # Remove ANY other camelCase variants that would cause extra fields.
        # Common mismatches we've seen: ownerUserId is accepted by Pydantic via alias,
        # but remove other camel keys that TemplateRead doesn't expect.
        for bad_key in ["createdAt", "updatedAt", "exampleNoteText", "sections_json", "sections_string", "sections_str"]:
            if bad_key in it:
                del it[bad_key]

        return it

    async def list_for_user(self, owner_user_id: str) -> List[TemplateRead]:
        items = await self.scan_all()
        filtered = [
            item
            for item in items
            if str(item.get("ownerUserId") or item.get("owner_user_id") or "") == owner_user_id
        ]

        normalized_items = [await self._normalize_item_for_read(item) for item in filtered]
        return [TemplateRead.model_validate(x) for x in normalized_items]

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
        normalized = await self._normalize_item_for_read(item)
        return TemplateRead.model_validate(normalized)

    async def update(self, template_id: str, payload: TemplateUpdate) -> TemplateRead:
        existing = await self.get_required_item(str(template_id))
        updates = payload.model_dump(exclude_unset=True)

        # Merge existing and updates, but keep raw existing keys to avoid losing other attributes
        merged = {**existing, **self.serialize_input(updates)}
        # Ensure updated_at uses snake_case timestamp (backend canonical)
        merged["updated_at"] = datetime.utcnow().isoformat()

        # Normalize before validating: handle camelCase leftovers and sections string
        normalized = await self._normalize_item_for_read(merged)

        # Persist normalized merged item (serialize_input will convert types)
        await run_in_thread(self.table.put_item, Item=self.serialize_input(normalized))

        # Return validated TemplateRead
        return TemplateRead.model_validate(normalized)

    async def delete(self, template_id: str) -> None:
        try:
            await run_in_thread(
                self.table.delete_item,
                Key={self.partition_key: str(template_id)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": self.partition_key},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"Template with id={template_id} was not found.") from exc
            raise


template_service = TemplateService(
    table_env_name="TEMPLATES_TABLE_NAME",
    default_table_name="medical-scribe-templates",
)