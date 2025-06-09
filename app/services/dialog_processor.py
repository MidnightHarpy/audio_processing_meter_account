from enum import Enum
from typing import Dict, Any, AsyncGenerator, Optional, AsyncIterator
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.audio import MeterReading
from app.ai.models.predict import Predictor
import re
import uuid
import logging
import os
import asyncio
from app.services.number_converter import NumberConverter
from app.services.asr_service import ASRService
from app.utils.crypto import cipher

logger = logging.getLogger(__name__)


class DialogProcessor:
    def __init__(self, db_session: Session, asr_service: ASRService):
        self.db = db_session
        self.asr_service = asr_service
        self.predictor = Predictor(os.path.abspath("app/ai/models/model_directory"))
        self.partial_results = []

    async def process_audio_background(self, content: bytes) -> Dict[str, Any]:
        try:
            # транскрибация в отдельном потоке
            audio_text = await asyncio.get_event_loop().run_in_executor(
                None,
                self.asr_service._sync_transcribe,
                content
            )

            processed_text = await asyncio.get_event_loop().run_in_executor(
                None,
                NumberConverter.convert,
                audio_text
            )

            return {
                "audio_text": audio_text,
                "processed_text": processed_text
            }

        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}")
            return {"error": str(e)}

    async def process_call(
            self,
            audio_stream: AsyncIterator[bytes],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        context = DialogContext(
            session_id=str(uuid.uuid4()),
            scales={1: ScaleData()}
        )

        async for audio_chunk in audio_stream:
            try:
                text = await self.asr_service.transcribe_async(audio_chunk)
                if text:
                    processed = NumberConverter.convert(text)
                    normalized = NumberConverter.normalize_numbers(processed)
                    self.partial_results.append(normalized)
                    yield self._process_partial(context)
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                yield self._build_error_response("Ошибка обработки аудио")

        yield await self._final_processing(context)

    async def _final_processing(self, context: DialogContext) -> Dict[str, Any]:
        full_text = " ".join(self.partial_results)
        entities = self.predictor.predict(full_text)
        account = next((e["text"] for e in entities if e["type"] == "ACCOUNT"), None)
        meter = next((e["text"] for e in entities if e["type"] == "METER"), None)

        if account and meter:
            return await self._handle_full_data(context, account, meter)
        return self._build_failed_response(reason="Не удалось распознать данные")

    async def _handle_full_data(self, context: DialogContext, account: str, meter: str) -> Dict[str, Any]:
        try:
            if not meter:
                raise ValueError("Значение показаний не найдены")

            clean_account = re.sub(r'\D', '', account)
            encrypted_account = cipher.encrypt(clean_account)
            clean_meter = re.sub(r'[^\d.]', '', meter)


            if len(clean_account) != 9:
                raise ValueError("Лицевой счет должен содержать 9 цифр")
            if len(clean_meter) > 5:
                raise ValueError("Показания счетчика не должны превышать 5 цифр")

            async with self.db.begin():
                result = await self.db.execute(
                    select(MeterReading)
                    .where(MeterReading.account_number == encrypted_account)
                    .order_by(MeterReading.timestamp.desc())
                    .limit(1)
                )
                last_reading = await result.scalars().first()

            if last_reading and int(clean_meter) < last_reading.scale1_value:
                return self._build_failed_response(
                    reason="Новые показания не могут быть меньше предыдущих"
                )

            # сохранение данных
            async with self.db.begin():
                meter_reading = MeterReading(
                    account_number=encrypted_account,
                    scale1_value=int(clean_meter),
                    session_id=context.session_id,
                    timestamp=datetime.now()
                )
                self.db.add(meter_reading)
                await self.db.commit()
                logger.info(f"Данные сохранены в БД: ID {meter_reading.id}")

        except ValueError as e:
            logger.error(f"Ошибка преобразования данных: {str(e)}")
            return self._build_failed_response(reason="Некорректный формат данных")
        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
            return self._build_failed_response(reason="Внутренняя ошибка системы")

