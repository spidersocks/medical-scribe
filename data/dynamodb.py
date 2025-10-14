import os
from functools import lru_cache
from typing import Any

import boto3


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")


@lru_cache(maxsize=1)
def get_session() -> boto3.session.Session:
    kwargs: dict[str, Any] = {"region_name": AWS_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    if AWS_SESSION_TOKEN:
        kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    return boto3.session.Session(**kwargs)


@lru_cache(maxsize=None)
def get_table(table_name_env: str, default_table_name: str):
    table_name = os.getenv(table_name_env, default_table_name)
    resource = get_session().resource("dynamodb")
    return resource.Table(table_name)