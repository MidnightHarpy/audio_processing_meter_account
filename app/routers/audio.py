from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.asr_service import ASRService
from app.services.dialog_processor import DialogProcessor
from app.utils.database import async_session
import logging
from contextlib import asynccontextmanager

router = APIRouter()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_db_session():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
