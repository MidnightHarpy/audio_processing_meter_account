from setuptools import setup, find_packages

setup(
    name="audio_processing",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart",
        "python-dotenv",
        "sqlalchemy[asyncio]>=1.4.0",
        "alembic",
        "asyncpg",
        "vosk",
        "noisereduce",
        "pydub",
        "python-jose[cryptography]>=3.3.0",
        "psycopg2-binary",
        "tensorflow-cpu>=2.15.0",
        "numpy",
        "joblib"
    ],
)