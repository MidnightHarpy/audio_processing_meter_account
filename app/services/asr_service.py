import os
import logging
from pathlib import Path
from vosk import Model, KaldiRecognizer
import wave
import json
from pydub import AudioSegment
import io
import tempfile
import asyncio
from typing import Optional
import numpy as np
import noisereduce as nr
import concurrent.futures

logger = logging.getLogger(__name__)
TRANSCODE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)
MAX_DURATION_MS = 300 * 1000

class ASRService:
    _model = None
    _recognizer = None
    _init_lock = asyncio.Lock()
    _initialized = False

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def initialize(cls):
        async with cls._init_lock:
            if not cls._initialized and cls._model is None:
                model_path = Path(os.getenv("VOSK_MODEL_PATH"))
                if not model_path.exists():
                    raise FileNotFoundError(f"Модель не найдена: {model_path}")

                logger.info("Загрузка модели Vosk...")
                try:
                    cls._model = Model(str(model_path))
                    cls._recognizer = KaldiRecognizer(
                        cls._model,
                        16000,
                        '["<unk>", "[noise]", "[laughter]"]'
                    )
                    cls._initialized = True
                    logger.info("Модель Vosk загружена")
                except Exception as e:
                    logger.critical(f"Ошибка загрузки модели: {str(e)}")
                    raise

    @classmethod
    async def cleanup(cls):
        async with cls._init_lock:
            cls._recognizer = None
            cls._model = None
            cls._initialized = False
            logger.info("Ресурсы ASRService освобождены")

    async def transcribe_async(self, content: bytes) -> Optional[str]:
        if not self._initialized:
            raise RuntimeError("ASRService не инициализирован")

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                TRANSCODE_EXECUTOR,
                self._sync_transcribe,
                content
            )
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {str(e)}")
            return None

    def _sync_transcribe(self, content: bytes) -> Optional[str]:
        try:
            converted_data = self._sync_convert_with_enhancement(content)
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                tmp.write(converted_data)
                tmp.seek(0)
                with wave.open(tmp.name, "rb") as wf:
                    if not self._validate_audio(wf):
                        logger.warning("Невалидное аудио после конвертации")
                        return ""

                    # Проверка длительности
                    duration = wf.getnframes() / wf.getframerate()
                    if duration > (MAX_DURATION_MS / 1000):
                        logger.error(f"Аудио слишком длинное: {duration:.1f}s > {MAX_DURATION_MS / 1000}s")
                        return ""

                    result = []
                    while True:
                        data = wf.readframes(4000)
                        if not data:
                            break
                        if self._recognizer.AcceptWaveform(data):
                            partial = json.loads(self._recognizer.Result())
                            result.append(partial.get("text", ""))
                    final = json.loads(self._recognizer.FinalResult())
                    result.append(final.get("text", ""))
                    return " ".join(filter(None, result)).strip()
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {str(e)}", exc_info=True)
            return None

    @staticmethod
    def _sync_convert_with_enhancement(audio_data: bytes) -> bytes:
        # конвертация аудио с улучшениями и дополнительными проверками
        try:
            # проверка минимальной длины аудио
            if len(audio_data) < 1024:
                raise ValueError("Audio file is too small (min 1KB)")

            # чтение аудио с обработкой исключений
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
            except Exception as e:
                audio = AudioSegment.from_raw(
                    io.BytesIO(audio_data),
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )

            # проверка и коррекция параметров
            audio = audio.set_channels(1).set_sample_width(2)
            if audio.frame_rate not in [8000, 16000]:
                audio = audio.set_frame_rate(16000)

            # обрезка до максимальной длины
            if len(audio) > MAX_DURATION_MS:
                audio = audio[:MAX_DURATION_MS]
                logger.warning(f"Audio truncated to {MAX_DURATION_MS}ms")

            # проверка на тишину
            samples = np.array(audio.get_array_of_samples())
            if np.all(samples == 0):
                raise ValueError("Audio contains only silence")

            # подавление шума и нормализация
            reduced_noise = nr.reduce_noise(
                y=samples,
                sr=audio.frame_rate,
                stationary=True,
                prop_decrease=0.85
            )
            processed = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=2,
                channels=1
            ).normalize(headroom=0.1)

            # экспорт в WAV
            buffer = io.BytesIO()
            processed.export(buffer, format="wav", codec="pcm_s16le")
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise RuntimeError(f"Audio processing error: {str(e)}")

    @staticmethod
    def _validate_audio(wav_file: wave.Wave_read) -> bool:
        valid_params = (
                wav_file.getnchannels() == 1 and
                wav_file.getsampwidth() == 2 and
                wav_file.getcomptype() == "NONE" and
                wav_file.getframerate() in [8000, 16000, 44100]
        )

        if not valid_params:
            logger.error(f"Неверные параметры аудио: {wav_file.getparams()}")

        return valid_params