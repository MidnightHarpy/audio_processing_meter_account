from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
import sys
import asyncio
from pathlib import Path
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncConnection

import logging

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from app.utils.database import Base

logger = logging.getLogger(__name__)
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations(connectable: AsyncConnection):
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

        inspector = await connection.run_sync(inspect)
        tables = inspector.get_table_names()
        logger.info(f"Existing tables: {tables}")


def run_migrations_online() -> None:
    connectable = create_engine(config.get_main_option("sqlalchemy.url"))

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

def do_run_migrations(connection):
    #Основная логика выполнения миграций
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        render_as_batch=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()