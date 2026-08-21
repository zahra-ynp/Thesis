"""Central environment-based configuration for notebooks and applications."""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "requirements.txt").exists() and (candidate / "modelling").exists():
            return candidate
    raise FileNotFoundError("Could not locate the thesis project root.")


PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is missing. Copy .env.example to .env and set the local value."
        )
    return value


def get_sqlalchemy_engine(*, fast_executemany: bool = False, pool_pre_ping: bool = True):
    """Create a SQL Server engine exclusively from environment configuration."""
    from sqlalchemy import create_engine

    server = required_env("DB_SERVER")
    database = required_env("DB_DATABASE")
    driver = required_env("DB_DRIVER")
    auth_mode = os.getenv("DB_AUTH_MODE", "trusted").strip().lower()
    trust_certificate = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "yes").strip()

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"TrustServerCertificate={trust_certificate}",
    ]
    if auth_mode == "trusted":
        parts.append("Trusted_Connection=yes")
    elif auth_mode == "sql":
        parts.extend([
            f"UID={required_env('DB_USERNAME')}",
            f"PWD={required_env('DB_PASSWORD')}",
        ])
    else:
        raise RuntimeError("DB_AUTH_MODE must be either 'trusted' or 'sql'.")

    connection_string = ";".join(parts) + ";"
    return create_engine(
        "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(connection_string),
        fast_executemany=fast_executemany,
        pool_pre_ping=pool_pre_ping,
    )
