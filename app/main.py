import asyncio
import json
import uuid
import logging
import os
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.asr_service import ASRService
from app.services.dialog_processor import DialogProcessor
from app.services.number_converter import NumberConverter
from app.models.audio import CallSession, MeterReading, FailedProcessingAttempt
from app.utils.crypto import cipher
from app.utils.database import SessionLocal
from app.utils.json_encoder import custom_json_encoder
from app.ai.models.predict import Predictor
from app.utils.database import Base
from app.routers import audio
from app.routers import health
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy import select, inspect, extract, and_
from sqlalchemy.sql import text
from sqlalchemy.orm import selectinload
from typing import Dict, Any
from contextlib import asynccontextmanager
from app.utils.database import engine
from app.ai.history.anomaly_detector import AnomalyDetector
from datetime import datetime
from typing import Optional
from fastapi import Query
from asyncio import Semaphore

CONCURRENT_LIMIT = 5
semaphore = Semaphore(CONCURRENT_LIMIT)

logger = logging.getLogger(__name__)

predictor = Predictor(
    model_path=os.path.abspath("app/ai/models/model_directory")
)

asr_service_instance: Optional[ASRService] = None
init_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_service_instance

    async with init_lock:
        try:
            await asyncio.sleep(15)

            # инициализация ASRService
            if asr_service_instance is None:
                asr_service_instance = ASRService()
                await asr_service_instance.initialize()
                logger.info("ASRService успешно инициализирован")

            # инициализация БД
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("БД инициализирован")

            # проверка работоспособности БД
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                logger.info("БД работоспособен")

                tables = await conn.run_sync(
                    lambda sync_conn: inspect(sync_conn).get_table_names()
                )
                if "meter_readings" not in tables:
                    logger.error("Таблица 'meter_readings' не найдена")
                    raise RuntimeError("Некорректная схема БД")


        except Exception as e:
            logger.critical(f"Ошибка инициализации: {str(e)}", exc_info=True)
            raise

    yield

    if asr_service_instance:
        await asr_service_instance.cleanup()
        logger.info("Ресурсы ASRService освобождены")

    if engine:
        await engine.dispose()


fastapi_app = FastAPI(
    lifespan=lifespan,
    default_response_class=JSONResponse
)

fastapi_app.json_encoder = custom_json_encoder
fastapi_app.include_router(audio.router, prefix="/api")
fastapi_app.include_router(health.router)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fastapi_app.log'),
        logging.StreamHandler()
    ]
)


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def process_audio_background(content: bytes, filename: str):
    detector = AnomalyDetector()
    async with semaphore:
        try:
            logger.info(f"Начало обработки файла: {filename}")

            async with SessionLocal() as db:
                # транскрибация
                audio_text = await asr_service_instance.transcribe_async(content)
                logger.info(f"VOSK Результат: '{audio_text}'")

                # конвертация чисел
                converter = NumberConverter()
                processed_text = converter.convert(audio_text)
                logger.info(f"Конвертация: '{processed_text}'")

                # извлечение сущностей
                processor = DialogProcessor(db, asr_service_instance)
                entities = await asyncio.get_event_loop().run_in_executor(
                    None, processor.predictor.predict, processed_text
                )
                logger.info(f"Извлеченные сущности: {json.dumps(entities, indent=2, ensure_ascii=False)}")

                # валидация данных
                account = next((e["text"] for e in entities if e["type"] == "ACCOUNT"), None)
                meter = next((e["text"] for e in entities if e["type"] == "METER"), None)

                # шифрование account_number
                encrypted_account = cipher.encrypt(account)

                # проверка длины лицевого счёта
                if not account or len(account) != 9 or not account.isdigit():
                    raise ValueError(f"Лицевой счёт должен содержать 9 цифр. Получено: {account}")

                # проверка показаний
                if not meter or len(meter) > 5 or not meter.isdigit():
                    raise ValueError(f"Некорректные показания: {meter}")

                if not account or not meter:
                    raise ValueError("Не удалось извлечь обязательные сущности (ACCOUNT и METER)")

                # получение истории показаний
                encrypted_account = cipher.encrypt(account)
                history_result = await db.execute(
                    select(MeterReading)
                    .where(MeterReading.account_number == encrypted_account)
                    .order_by(MeterReading.timestamp.desc())
                    .limit(12)
                )
                history = history_result.scalars().all()

                # проверка аномалий
                detector = AnomalyDetector()
                anomaly_data = await detector.detect(history, int(meter))

                # сохранение записи
                reading = MeterReading(
                    account_number=encrypted_account,
                    scale1_value=int(meter),
                    timestamp=datetime.now(),
                    is_anomaly=anomaly_data['is_anomaly'],
                    deviation_score=anomaly_data['deviation'],
                    historical_avg=anomaly_data['historical_avg'],
                    expected_range=anomaly_data['expected_range'],
                )

                db.add(reading)
                await db.commit()
                logger.info(f"Успешно сохранено: ID {reading.id}")

        except Exception as e:
            logger.error(logger.error(f"Ошибка обработки файла {filename}: {str(e)}"), exc_info=True)
            async with SessionLocal() as db:
                attempt = FailedProcessingAttempt(
                    original_audio_path=filename,
                    processed_text=processed_text if 'processed_text' in locals() else "",
                    raw_text=audio_text if 'audio_text' in locals() else "",
                    error_reason=str(e),
                    created_at=datetime.now()
                )
                db.add(attempt)
                await db.commit()
                logger.error(f"Сохранена неудачная попытка обработки: ID {attempt.id}")

fastapi_app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@fastapi_app.post("/test_audio")
async def test_audio(file: UploadFile = File(...)):
    logger.info(f"Начало обработки файла: {file.filename}")

    try:
        if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.aac')):
            raise HTTPException(400, detail="Неподдерживаемый формат аудио")

        content = await file.read()
        async with SessionLocal() as db:
            audio_text = await asr_service_instance.transcribe_async(content)
            if not audio_text:
                raise HTTPException(422, detail="Не удалось распознать речь")

            converter = NumberConverter()
            processed_text = converter.convert(audio_text)

            processor = DialogProcessor(db, asr_service_instance)
            entities = processor.predictor.predict(processed_text)

            account = next((e["text"] for e in entities if e["type"] == "ACCOUNT"), None)
            meter = next((e["text"] for e in entities if e["type"] == "METER"), None)

            encrypted_account = cipher.encrypt(account)

            if not account or len(account) != 9 or not account.isdigit():
                raise ValueError("Лицевой счет должен содержать 9 цифр")

            if not meter or len(meter) > 5 or not meter.isdigit():
                raise ValueError("Некорректные показания счетчика")

            session = CallSession(
                session_id=str(uuid.uuid4()),
                account_number=cipher.encrypt(account),
                start_time=datetime.now(),
                status="COMPLETED"
            )

            reading = MeterReading(
                account_number=encrypted_account,
                scale1_value=int(meter),
                session_id=session.session_id,
                timestamp=datetime.now()
            )

            db.add(session)
            db.add(reading)
            await db.commit()

            return JSONResponse({
                "message": "Данные успешно сохранены",
                "account": account,
                "meter": meter,
                "session_id": str(session.session_id)
            })

    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        async with SessionLocal() as db:
            attempt = FailedProcessingAttempt(
                original_audio_path=file.filename,
                processed_text=processed_text if 'processed_text' in locals() else '',
                raw_text=audio_text,
                error_reason=str(e),
                id=str(uuid.uuid4())
            )
            db.add(attempt)
            await db.commit()
        raise HTTPException(500, detail=f"Ошибка обработки. ID ошибки: {attempt.id}")


@fastapi_app.post("/confirm")
async def confirm_data(data: Dict[str, Any]):
    async with SessionLocal() as db:
        try:
            session_id = data.get("session_id")
            confirmation = data.get("confirmation", "").lower()

            result = await db.execute(
                select(CallSession)
                .filter_by(session_id=session_id)
                .options(selectinload(CallSession.readings))
            )
            session = result.scalars().first()

            if not session:
                raise HTTPException(404, detail="Сессия не найдена")

            if confirmation.lower() == "да":
                session.status = "COMPLETED"
            else:
                session.status = "FAILED"

            await db.commit()
            return {"status": session.status}

        except Exception as e:
            await db.rollback()
            logger.error(f"Ошибка подтверждения: {str(e)}", exc_info=True)
            raise HTTPException(500, detail=str(e))


@fastapi_app.get("/session/{session_id}")
async def get_session(session_id: str):
    async with SessionLocal() as db:
        try:
            result = await db.execute(
                select(CallSession)
                .options(selectinload(CallSession.readings))
                .filter_by(session_id=session_id)
            )
            session = result.scalars().first()

            if not session:
                raise HTTPException(404, detail="Сессия не найдена")

            return {
                "session_id": session_id,
                "status": session.status,
                "account_number": session.account_number,
                "readings": [{
                    "timestamp": r.timestamp,
                    "value": r.scale1_value
                } for r in session.readings]
            }
        except Exception as e:
            logging.error(f"Ошибка получения сессии: {e}", exc_info=True)
            raise HTTPException(500, detail=str(e))

@fastapi_app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "menu": [
            {"title": "Главная", "url": "/"},
            {"title": "История", "url": "/history"},
            {"title": "Загрузить", "url": "/upload"}
        ]
    })

@fastapi_app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@fastapi_app.get("/records", response_class=HTMLResponse)
async def manage_records(
        request: Request,
        account_number: Optional[str] = Query(None),
        year: Optional[str] = Query(None),
        month: Optional[str] = Query(None),
        anomalies_only: bool = Query(False),
        page: int = Query(1, ge=1),
        per_page: int = Query(100, ge=1, le=200)
):
    async with SessionLocal() as db:
        try:

            # валидация параметров
            encrypted_account = None
            account_number = account_number.strip() if account_number else None
            page = max(page, 1)
            per_page = min(per_page, 200)
            offset = (page - 1) * per_page

            # Преобразование параметров фильтрации
            year_int = int(year) if year and year.isdigit() else None
            month_int = int(month) if month and month.isdigit() else None

            # условия фильтрации
            filters = []

            if account_number:
                if len(account_number) != 9 or not account_number.isdigit():
                    raise HTTPException(400, detail="Некорректный формат лицевого счета (9 цифр)")

                try:
                    # шифруем номер перед поиском
                    encrypted_account = cipher.encrypt(account_number.strip())
                    filters.append(MeterReading.account_number == encrypted_account)
                    logger.debug(f"Зашифрованный номер для поиска: {encrypted_account}")
                    filters.append(MeterReading.account_number == encrypted_account)
                except Exception as e:
                    logger.error(f"Ошибка шифрования: {str(e)}")
                    raise HTTPException(500, detail="Ошибка обработки данных")

            query = select(MeterReading).order_by(MeterReading.timestamp.desc())

            if encrypted_account:
                filters.append(MeterReading.account_number == encrypted_account)

            if year_int:
                filters.append(extract('year', MeterReading.timestamp) == year_int)
            if month_int:
                filters.append(extract('month', MeterReading.timestamp) == month_int)
            if anomalies_only:
                filters.append(MeterReading.is_anomaly == True)

            # применение фильтров
            if filters:
                query = query.where(and_(*filters))

            # пагинация
            result = await db.execute(query.offset(offset).limit(per_page))
            records = result.scalars().all()
            has_next = len(records) == per_page

            # расшифровка записей
            decrypted_records = []
            for record in records:
                try:
                    decrypted_account = cipher.decrypt(record.account_number)
                except Exception as e:
                    logger.error(f"Ошибка дешифровки счета: {str(e)}")
                    decrypted_account = "ОШИБКА ДЕШИФРОВКИ"

                decrypted_records.append({
                    "id": record.id,
                    "account_number": decrypted_account,
                    "scale1_value": record.scale1_value,
                    "timestamp": record.timestamp,
                    "is_anomaly": record.is_anomaly,
                    "deviation_score": record.deviation_score,
                    "historical_avg": record.historical_avg,
                    "expected_range": record.expected_range
                })

            # получение данных для фильтров
            years = await get_available_years(db)
            months = [
                (1, "Январь"), (2, "Февраль"), (3, "Март"),
                (4, "Апрель"), (5, "Май"), (6, "Июнь"),
                (7, "Июль"), (8, "Август"), (9, "Сентябрь"),
                (10, "Октябрь"), (11, "Ноябрь"), (12, "Декабрь")
            ]

            return templates.TemplateResponse("records.html", {
                "request": request,
                "records": decrypted_records,
                "years": years,
                "months": months,
                "selected_year": year_int,
                "selected_month": month_int,
                "page": page,
                "has_next": has_next,
                "current_filters": {
                    "account_number": account_number or "",
                    "year": year or "",
                    "month": month or "",
                    "anomalies_only": anomalies_only
                }
            })

        except ValueError as e:
            logger.error(f"Ошибка параметров: {str(e)}")
            raise HTTPException(400, detail="Некорректные параметры фильтрации")
        except Exception as e:
            logger.error(f"Ошибка получения записей: {str(e)}", exc_info=True)
            raise HTTPException(500, detail="Ошибка базы данных")


async def get_available_years(db: AsyncSession):
    result = await db.execute(
        select(extract('year', MeterReading.timestamp).distinct())
        .where(MeterReading.timestamp.isnot(None))
        .order_by(extract('year', MeterReading.timestamp).desc())
    )
    return [int(row[0]) for row in result.all() if row[0] is not None]


@fastapi_app.get("/edit/{record_id}", response_class=HTMLResponse)
async def edit_record_page(request: Request, record_id: str):
    async with SessionLocal() as db:
        record = await db.get(MeterReading, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Запись не найдена")

        # получаем полную информацию о попытке обработки
        attempt = await db.execute(
            select(FailedProcessingAttempt)
            .order_by(FailedProcessingAttempt.created_at.desc())
        )
        attempt = attempt.scalar_one_or_none()

        return templates.TemplateResponse("edit.html", {
            "request": request,
            "record": record,
            "raw_text": attempt.raw_text if attempt else "Не найдено",
            "processed_text": attempt.processed_text if attempt else "Не найдено"
        })

@fastapi_app.post("/update/{record_id}")
async def update_record(
    record_id: str,
    account: str = Form(...),
    meter: str = Form(...)
):
    async with SessionLocal() as db:
        try:
            # Получаем запись
            record = await db.get(MeterReading, record_id)
            if not record:
                raise HTTPException(404, detail="Запись не найдена")

            # Валидация
            if len(account) != 9 or not account.isdigit():
                raise HTTPException(400, detail="Некорректный номер счета")

            if not meter.isdigit() or len(meter) > 5:
                raise HTTPException(400, detail="Некорректные показания")

            record.account_number = cipher.encrypt(account)
            record.scale1_value = int(meter)

            await db.commit()
            await db.refresh(record)

            return RedirectResponse(url="/records", status_code=303)

        except Exception as e:
            await db.rollback()
            logger.error(f"Ошибка обновления: {str(e)}")
            raise HTTPException(500, detail="Ошибка сервера")


@fastapi_app.post("/delete/{record_id}")
async def delete_record(record_id: str):
    async with SessionLocal() as db:
        try:
            record = await db.get(MeterReading, record_id)
            if not record:
                raise HTTPException(404, detail="Запись не найдена")

            await db.delete(record)
            await db.commit()
            return JSONResponse({"status": "success", "message": "Record deleted"})


        except Exception as e:
            await db.rollback()
            logger.error(f"Ошибка удаления: {str(e)}", exc_info=True)
            raise HTTPException(500, detail=str(e))


@fastapi_app.get("/test_conversion")
async def test_conversion(input_text: str):
    converter = NumberConverter()
    converted = converter.convert(input_text)

    account_valid = any(c.isdigit() for c in converted) and len(converted) == 9
    meter_valid = any(c.isdigit() for c in converted) and 1 <= len(converted) <= 5

    return {
        "original": input_text,
        "converted": converted,
        "validation": {
            "account": account_valid,
            "meter": meter_valid
        }
    }


@fastapi_app.get("/new", response_class=HTMLResponse)
async def new_readings_page(request: Request):
    return templates.TemplateResponse("new_readings.html", {"request": request})


@fastapi_app.get("/failed", response_class=HTMLResponse)
async def failed_readings_page(
        request: Request,
        error_reason: Optional[str] = None,
        attempt_id: Optional[str] = None
):
    async with SessionLocal() as db:
        query = select(FailedProcessingAttempt).order_by(FailedProcessingAttempt.created_at.desc())

        if error_reason:
            query = query.where(FailedProcessingAttempt.error_reason.ilike(f"%{error_reason}%"))
        if attempt_id:
            query = query.where(FailedProcessingAttempt.id == attempt_id)

        attempts = await db.execute(query)
        return templates.TemplateResponse("failed_readings.html", {
            "request": request,
            "attempts": attempts.scalars().all()
        })


@fastapi_app.post("/retry/{attempt_id}")
async def retry_attempt(
        attempt_id: str,
        account: str = Form(None),
        meter: str = Form(None)
):
    async with SessionLocal() as db:
        attempt = await db.get(FailedProcessingAttempt, attempt_id)
        if not attempt:
            raise HTTPException(status_code=404, detail="Попытка не найдена")

        # извлечение сущностей через модель
        entities = predictor.predict(attempt.processed_text)

        final_account = account or next(
            (e["text"] for e in entities if e["type"] == "ACCOUNT"),
            ""
        )
        final_meter = meter or next(
            (e["text"] for e in entities if e["type"] == "METER"),
            ""
        )

        # валидация через прямые проверки
        if not final_account and not final_meter:
            raise HTTPException(400, detail="Необходимо указать хотя бы одно значение")

        if final_account and (len(final_account) != 9 or not final_account.isdigit()):
            raise HTTPException(400, detail="Некорректный формат лицевого счета")

        if final_meter and (not final_meter.isdigit() or len(final_meter) > 5):
            raise HTTPException(400, detail="Некорректные показания счетчика")

        # создание новой записи
        new_reading = MeterReading(
            account_number=final_account,
            scale1_value=int(final_meter) if final_meter else 0,
            timestamp=datetime.now()
        )

        db.add(new_reading)
        await db.delete(attempt)
        await db.commit()

        return RedirectResponse(url="/records", status_code=303)

@fastapi_app.post("/update_attempt/{attempt_id}")
async def update_attempt(attempt_id: str, corrected_text: str = Body(...)):
    async with SessionLocal() as db:
        attempt = await db.get(FailedProcessingAttempt, attempt_id)
        attempt.processed_text = corrected_text
        await db.commit()
        return {"status": "updated"}


@fastapi_app.post("/upload")
async def upload_audio(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # Проверка формата файла
        if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.aac')):
            raise HTTPException(400, detail="Неподдерживаемый формат аудио")

        # Чтение содержимого файла
        content = await file.read()

        # Создаем задачу для обработки
        background_tasks.add_task(
            process_audio_background,
            content,
            file.filename
        )

        return JSONResponse({
            "status": "processing",
            "message": "Файл принят в обработку",
            "filename": file.filename,
            "received_at": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Ошибка загрузки: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@fastapi_app.post("/delete_attempt/{attempt_id}")
async def delete_attempt(attempt_id: str):
    async with SessionLocal() as db:
        attempt = await db.get(FailedProcessingAttempt, attempt_id)
        if not attempt:
            raise HTTPException(404, detail="Попытка не найдена")

        await db.delete(attempt)
        await db.commit()
        return RedirectResponse(url="/failed", status_code=303)


@fastapi_app.post("/update_reading/{reading_id}")
async def update_reading(
        reading_id: str,
        account: str = Form(...),
        meter: str = Form(...)
):
    async with SessionLocal() as db:
        reading = await db.get(MeterReading, reading_id)
        if not reading:
            raise HTTPException(status_code=404, detail="Запись не найдена")

        if len(account) != 9 or not account.isdigit():
            raise HTTPException(400, detail="Некорректный номер счета")

        if not meter.isdigit() or len(meter) > 5:
            raise HTTPException(400, detail="Некорректные показания")

        reading.account_number = cipher.encrypt(account)
        reading.scale1_value = int(meter)
        await db.commit()
        return RedirectResponse(url="/records", status_code=303)


@fastapi_app.post("/retry_attempt/{attempt_id}")
async def retry_attempt(
    attempt_id: str,
    account: str = Form(...),
    meter: str = Form(...)
):
    async with SessionLocal() as db:
        if len(account) != 9 or not account.isdigit():
            raise HTTPException(400, detail="Лицевой счет должен содержать 9 цифр")
        if not meter.isdigit() or len(meter) > 5:
            raise HTTPException(400, detail="Некорректные показания")

        attempt = await db.get(FailedProcessingAttempt, attempt_id)
        if not attempt:
            raise HTTPException(404, detail="Попытка не найдена")

        encrypted_account = cipher.encrypt(account)

        # создаем новую запись в MeterReading
        new_reading = MeterReading(
            account_number=encrypted_account,
            scale1_value=int(meter),
            timestamp=datetime.now()
        )
        db.add(new_reading)

        # удаляем попытку из failed_attempts
        await db.delete(attempt)
        await db.commit()

    return RedirectResponse(url="/records", status_code=303)


@fastapi_app.get("/correct/{attempt_id}", response_class=HTMLResponse)
async def correct_attempt_page(
    request: Request,
    attempt_id: str
):
    async with SessionLocal() as db:
        attempt = await db.get(FailedProcessingAttempt, attempt_id)
        if not attempt:
            raise HTTPException(status_code=404, detail="Попытка не найдена")

        if not attempt.processed_text:
            attempt.processed_text = ""

        try:
            entities = predictor.predict(attempt.processed_text)
        except Exception as e:
            logger.error(f"Ошибка извлечения сущностей: {str(e)}")
            entities = []

        account = next((e["text"] for e in entities if e["type"] == "ACCOUNT"), "")
        meter = next((e["text"] for e in entities if e["type"] == "METER"), "")

        return templates.TemplateResponse("correct_attempt.html", {
            "request": request,
            "attempt": attempt,
            "attempt_id": attempt_id,
            "account": account,
            "meter": meter
        })


@fastapi_app.post("/delete_failed/{attempt_id}")
async def delete_failed_attempt(attempt_id: str):
    async with SessionLocal() as db:
        try:
            attempt = await db.get(FailedProcessingAttempt, attempt_id)
            if not attempt:
                raise HTTPException(status_code=404, detail="Попытка не найдена")

            await db.delete(attempt)
            await db.commit()
            return RedirectResponse(url="/failed", status_code=303)

        except Exception as e:
            await db.rollback()
            logger.error(f"Ошибка удаления: {str(e)}", exc_info=True)
            raise HTTPException(500, detail=str(e))