from __future__ import annotations

from typing import List
from uuid import UUID, uuid4

from botocore.exceptions import ClientError  # type: ignore

from services.base import DynamoServiceMixin, run_in_thread
from services.exceptions import NotFoundError
from schemas.user import UserCreate, UserRead, UserUpdate


class UserService(DynamoServiceMixin):
    async def create(self, payload: UserCreate) -> UserRead:
        data = payload.model_dump()
        raw_id = data.pop("id", None)
        user_id = str(raw_id or uuid4())
        item = {"id": user_id, **self.serialize_input(data)}
        await run_in_thread(self.table.put_item, Item=item)
        clean_item = self.clean(item)
        return UserRead.model_validate(clean_item)

    async def list(self, limit: int = 100, offset: int = 0) -> List[UserRead]:
        items = await self.scan_all()
        paginated = items[offset : offset + limit]
        return [UserRead.model_validate(item) for item in paginated]

    async def get(self, user_id: str | UUID) -> UserRead:
        item = await self.get_required_item(str(user_id))
        return UserRead.model_validate(item)

    async def update(self, user_id: str | UUID, payload: UserUpdate) -> UserRead:
        existing = await self.get_required_item(str(user_id))
        updates = payload.model_dump(exclude_unset=True)
        merged = {**existing, **self.serialize_input(updates)}
        await run_in_thread(self.table.put_item, Item=self.serialize_input(merged))
        clean_item = self.clean(merged)
        return UserRead.model_validate(clean_item)

    async def delete(self, user_id: str | UUID) -> None:
        try:
            await run_in_thread(
                self.table.delete_item,
                Key={self.partition_key: str(user_id)},
                ConditionExpression="attribute_exists(#pk)",
                ExpressionAttributeNames={"#pk": self.partition_key},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise NotFoundError(f"User with id={user_id} was not found.") from exc
            raise


user_service = UserService(
    table_env_name="USERS_TABLE_NAME",
    default_table_name="medical-scribe-users",
)