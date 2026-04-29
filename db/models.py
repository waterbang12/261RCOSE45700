import os
import aiosqlite
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "golf.db")


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                phone       TEXT UNIQUE NOT NULL,
                name        TEXT,
                visit_count INTEGER DEFAULT 0,
                pref_temp   REAL DEFAULT 24.0,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS visits (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER REFERENCES customers(id),
                bay_id      TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                ended_at    TEXT,
                swing_count INTEGER DEFAULT 0,
                duration_min REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                bay_id          TEXT NOT NULL,
                event_type      TEXT NOT NULL,
                severity        TEXT NOT NULL,
                confidence      REAL,
                evidence        TEXT,
                alerted         INTEGER DEFAULT 0,
                occurred_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS env_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                bay_id      TEXT NOT NULL,
                temperature REAL,
                humidity    REAL,
                action      TEXT,
                occurred_at TEXT NOT NULL
            );
        """)
        await db.commit()


# ── 고객 ──────────────────────────────────────────────────────────────────────

async def upsert_customer(phone: str, name: str = "") -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            """
            INSERT INTO customers (phone, name, visit_count, created_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(phone) DO UPDATE SET
                visit_count = visit_count + 1,
                name = COALESCE(NULLIF(excluded.name, ''), customers.name)
            """,
            (phone, name, datetime.now().isoformat()),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM customers WHERE phone = ?", (phone,))
        row = await cursor.fetchone()
        return dict(row)


async def get_customer(phone: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM customers WHERE phone = ?", (phone,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_pref_temp(phone: str, temp: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE customers SET pref_temp = ? WHERE phone = ?", (temp, phone)
        )
        await db.commit()


# ── 방문 ──────────────────────────────────────────────────────────────────────

async def start_visit(customer_id: int, bay_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO visits (customer_id, bay_id, started_at) VALUES (?, ?, ?)",
            (customer_id, bay_id, datetime.now().isoformat()),
        )
        await db.commit()
        return cursor.lastrowid


async def end_visit(visit_id: int, swing_count: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE visits SET ended_at = ?, swing_count = ?,
                duration_min = ROUND(
                    (JULIANDAY(?) - JULIANDAY(started_at)) * 1440, 1
                )
            WHERE id = ?
            """,
            (datetime.now().isoformat(), swing_count, datetime.now().isoformat(), visit_id),
        )
        await db.commit()


# ── 이벤트 로그 ───────────────────────────────────────────────────────────────

async def log_event(
    bay_id: str,
    event_type: str,
    severity: str,
    confidence: float,
    evidence: str,
    alerted: bool,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO events (bay_id, event_type, severity, confidence, evidence, alerted, occurred_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (bay_id, event_type, severity, confidence, evidence, int(alerted), datetime.now().isoformat()),
        )
        await db.commit()


async def get_today_events() -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM events WHERE occurred_at LIKE ? ORDER BY occurred_at",
            (f"{today}%",),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


# ── 환경 로그 ─────────────────────────────────────────────────────────────────

async def log_env(bay_id: str, temperature: float, humidity: float, action: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO env_logs (bay_id, temperature, humidity, action, occurred_at) VALUES (?, ?, ?, ?, ?)",
            (bay_id, temperature, humidity, action, datetime.now().isoformat()),
        )
        await db.commit()


async def get_today_env_logs(bay_id: str) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM env_logs WHERE bay_id = ? AND occurred_at LIKE ? ORDER BY occurred_at",
            (bay_id, f"{today}%"),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
