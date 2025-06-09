from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, func, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID
from app.utils.database import Base
import uuid
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey


class MeterReading(Base):
    __tablename__ = "meter_readings"


    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_number = Column(String, index=True)
    timestamp = Column(DateTime)
    scale1_value = Column(Integer)
    is_anomaly = Column(Boolean, default=False, index=True)
    deviation_score = Column(Float)
    historical_avg = Column(Float)
    expected_range = Column(String(50))


class FailedProcessingAttempt(Base):
    __tablename__ = "failed_attempts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_audio_path = Column(String)
    processed_text = Column(Text, default='')
    raw_text = Column(Text, default='')
    error_reason = Column(String)
    raw_entities = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    retry_count = Column(Integer, default=0)


