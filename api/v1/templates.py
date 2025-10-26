from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Response, status, Depends

from api.deps import guard_service, get_current_user_id  # assume you have a way to get owner from auth
from schemas.template import TemplateCreate, TemplateRead, TemplateUpdate
from services import template_service

router = APIRouter()


@router.get("/", response_model=List[TemplateRead])
async def list_templates(user_id: UUID):
    # This route mirrors list_patients/list_consultations style: user_id passed in query
    return await guard_service(template_service.list_for_user(str(user_id)))


@router.post("/", response_model=TemplateRead, status_code=status.HTTP_201_CREATED)
async def create_template(user_id: UUID, payload: TemplateCreate):
    return await guard_service(template_service.create(str(user_id), payload))


@router.get("/{template_id}", response_model=TemplateRead)
async def get_template(template_id: UUID):
    return await guard_service(template_service.get(str(template_id)))


@router.patch("/{template_id}", response_model=TemplateRead)
async def update_template(template_id: UUID, payload: TemplateUpdate):
    return await guard_service(template_service.update(str(template_id), payload))


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(template_id: UUID):
    await guard_service(template_service.delete(str(template_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)