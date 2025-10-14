import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, TypeVar
from uuid import UUID

from boto3.dynamodb.conditions import Key  # type: ignore

from services.exceptions import NotFoundError, ValidationError

T = TypeVar("T")


def serialize_for_dynamo(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, list):
        return [serialize_for_dynamo(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_for_dynamo(val) for key, val in value.items()}
    return value


def clean_from_dynamo(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    if isinstance(value, list):
        return [clean_from_dynamo(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_from_dynamo(val) for key, val in value.items()}
    return value


def run_in_thread(func: Callable[..., T], *args, **kwargs) -> Coroutine[Any, Any, T]:
    return asyncio.to_thread(func, *args, **kwargs)


@dataclass
class DynamoServiceMixin:
    table_env_name: str
    default_table_name: str
    partition_key: str = "id"

    @property
    def table(self):
        from data.dynamodb import get_table  # local import to avoid circular deps

        return get_table(self.table_env_name, self.default_table_name)

    def serialize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {key: serialize_for_dynamo(value) for key, value in data.items()}

    def clean(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {key: clean_from_dynamo(value) for key, value in item.items()}

    async def scan_all(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        response = await run_in_thread(self.table.scan)
        items.extend(response.get("Items", []))

        while "LastEvaluatedKey" in response:
            response = await run_in_thread(
                self.table.scan,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))

        return [self.clean(item) for item in items]

    async def get_required_item(self, item_id: str) -> Dict[str, Any]:
        response = await run_in_thread(
            self.table.get_item,
            Key={self.partition_key: item_id},
        )
        item = response.get("Item")
        if not item:
            raise NotFoundError(f"{self.__class__.__name__} with id={item_id} was not found.")
        return self.clean(item)

    async def ensure_unique(self, index_name: str, value: str, message: str) -> None:
        response = await run_in_thread(
            self.table.query,
            IndexName=index_name,
            KeyConditionExpression=Key(index_name).eq(value),  # type: ignore[arg-type]
            Limit=1,
        )
        if response.get("Items"):
            raise ValidationError(message)