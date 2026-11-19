
from pathlib import Path
import sqlite3
import json
import uuid

from datetime import datetime, timezone


# =========================================
# DATABASE CONFIGURATION
# =========================================

DB_DIR = Path("data")
DB_PATH = DB_DIR / "railresolve.db"


def current_timestamp():
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    """Create SQLite database connection."""

    DB_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# =========================================
# DATABASE INITIALIZATION
# =========================================

def init_db():
    """Create tables and migrate existing database."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Main complaint table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                complaint_text TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'Unassessed',
                status TEXT NOT NULL DEFAULT 'Open',

                train_number TEXT,
                journey_date TEXT,
                location TEXT,

                detected_information TEXT NOT NULL DEFAULT '{}',
                missing_information TEXT NOT NULL DEFAULT '[]',

                response_draft TEXT,
                evidence TEXT NOT NULL DEFAULT '[]',

                staff_review_status TEXT NOT NULL DEFAULT 'Pending',
                staff_notes TEXT,
                staff_reviewed_by TEXT,
                staff_reviewed_at TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Migrate older databases without losing existing records.
        existing_columns = {
            row["name"]
            for row in cursor.execute(
                "PRAGMA table_info(cases)"
            ).fetchall()
        }

        migrations = {
            "staff_reviewed_by": "TEXT",
            "staff_reviewed_at": "TEXT",
        }

        for column, data_type in migrations.items():

            if column not in existing_columns:

                cursor.execute(
                    f"ALTER TABLE cases ADD COLUMN {column} {data_type}"
                )

        # Case status history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                notes TEXT,
                changed_at TEXT NOT NULL,

                FOREIGN KEY (case_id)
                    REFERENCES cases(case_id)
                    ON DELETE CASCADE
            )
        """)

        # Staff review audit history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_history (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,

                decision TEXT NOT NULL
                    CHECK(decision IN ('Approved', 'Rejected')),

                reviewed_by TEXT NOT NULL,
                comments TEXT,
                reviewed_at TEXT NOT NULL,
                response_draft_snapshot TEXT,

                FOREIGN KEY (case_id)
                    REFERENCES cases(case_id)
                    ON DELETE CASCADE
            )
        """)

        # Database indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cases_category
            ON cases(category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cases_status
            ON cases(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_review_history_case
            ON review_history(case_id)
        """)

        connection.commit()

        print("SQLite database initialized successfully!")
        print("Database location:", DB_PATH)

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# =========================================
# CREATE A NEW COMPLAINT CASE
# =========================================

def create_case(
    complaint_text,
    category,
    status="Open",
    priority="Unassessed",
    train_number=None,
    journey_date=None,
    location=None,
    detected_information=None,
    missing_information=None,
    response_draft=None,
    evidence=None,
):
    """Create a new complaint case."""

    if not isinstance(complaint_text, str) or not complaint_text.strip():
        raise ValueError("Complaint text cannot be empty.")

    if not category or not category.strip():
        raise ValueError("Complaint category is required.")

    case_id = (
        "RR-"
        + datetime.now(timezone.utc).strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:8].upper()
    )

    timestamp = current_timestamp()

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO cases (
                case_id,
                complaint_text,
                category,
                priority,
                status,
                train_number,
                journey_date,
                location,
                detected_information,
                missing_information,
                response_draft,
                evidence,
                staff_review_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            complaint_text.strip(),
            category.strip(),
            priority,
            status,
            train_number,
            journey_date,
            location,
            json.dumps(
                detected_information or {},
                ensure_ascii=False
            ),
            json.dumps(
                missing_information or [],
                ensure_ascii=False
            ),
            response_draft,
            json.dumps(
                evidence or [],
                ensure_ascii=False
            ),
            "Pending",
            timestamp,
            timestamp,
        ))

        # Initial status history entry
        cursor.execute("""
            INSERT INTO case_history (
                case_id,
                previous_status,
                new_status,
                changed_by,
                notes,
                changed_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            None,
            status,
            "RailResolve AI",
            "Complaint case created.",
            timestamp,
        ))

        connection.commit()

        return case_id

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# =========================================
# RETRIEVE ONE CASE
# =========================================

def get_case(case_id):
    """Retrieve a case using its unique ID."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM cases
            WHERE case_id = ?
        """, (case_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        result = dict(row)

        for field in [
            "detected_information",
            "missing_information",
            "evidence",
        ]:
            result[field] = json.loads(
                result.get(field) or (
                    "{}" if field == "detected_information"
                    else "[]"
                )
            )

        return result

    finally:
        connection.close()


# =========================================
# LIST CASES
# =========================================

def list_cases(status=None, category=None, limit=50):
    """List complaints with optional filters."""

    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Limit must be a positive integer.")

    query = "SELECT * FROM cases WHERE 1=1"
    parameters = []

    if status:
        query += " AND status = ?"
        parameters.append(status)

    if category:
        query += " AND category = ?"
        parameters.append(category)

    query += " ORDER BY created_at DESC LIMIT ?"
    parameters.append(limit)

    connection = get_connection()

    try:
        cursor = connection.cursor()
        cursor.execute(query, parameters)

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:
        connection.close()


# =========================================
# UPDATE CASE STATUS
# =========================================

def update_case_status(
    case_id,
    new_status,
    changed_by="Staff",
    notes=None,
):
    """Update case status and record its history."""

    if not new_status or not new_status.strip():
        raise ValueError("New status cannot be empty.")

    if not changed_by or not changed_by.strip():
        raise ValueError("Staff name is required.")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT status
            FROM cases
            WHERE case_id = ?
        """, (case_id,))

        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Case not found: {case_id}")

        previous_status = row["status"]
        timestamp = current_timestamp()

        cursor.execute("""
            UPDATE cases
            SET status = ?,
                updated_at = ?
            WHERE case_id = ?
        """, (
            new_status.strip(),
            timestamp,
            case_id,
        ))

        cursor.execute("""
            INSERT INTO case_history (
                case_id,
                previous_status,
                new_status,
                changed_by,
                notes,
                changed_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            previous_status,
            new_status.strip(),
            changed_by.strip(),
            notes,
            timestamp,
        ))

        connection.commit()

        print("Case status updated successfully!")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# =========================================
# GET CASE STATUS HISTORY
# =========================================

def get_case_history(case_id):
    """Retrieve status changes for a case."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM case_history
            WHERE case_id = ?
            ORDER BY changed_at ASC
        """, (case_id,))

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:
        connection.close()


# =========================================
# STAFF REVIEW: APPROVE OR REJECT
# =========================================

def review_case(
    case_id,
    decision,
    reviewed_by,
    comments=None,
):
    """
    Record approval or rejection of an AI response draft.

    This does not resolve a railway complaint or approve a refund.
    """

    if decision not in ("Approved", "Rejected"):
        raise ValueError(
            "Decision must be Approved or Rejected."
        )

    if not reviewed_by or not reviewed_by.strip():
        raise ValueError("Reviewer name is required.")

    comments = (comments or "").strip()

    if decision == "Rejected" and not comments:
        raise ValueError(
            "Comments are required when rejecting a draft."
        )

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT response_draft
            FROM cases
            WHERE case_id = ?
        """, (case_id,))

        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Case not found: {case_id}")

        draft = row["response_draft"]

        if not draft or not draft.strip():
            raise ValueError(
                "This case has no response draft to review."
            )

        timestamp = current_timestamp()

        cursor.execute("""
            UPDATE cases
            SET staff_review_status = ?,
                staff_notes = ?,
                staff_reviewed_by = ?,
                staff_reviewed_at = ?,
                updated_at = ?
            WHERE case_id = ?
        """, (
            decision,
            comments,
            reviewed_by.strip(),
            timestamp,
            timestamp,
            case_id,
        ))

        cursor.execute("""
            INSERT INTO review_history (
                case_id,
                decision,
                reviewed_by,
                comments,
                reviewed_at,
                response_draft_snapshot
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            decision,
            reviewed_by.strip(),
            comments,
            timestamp,
            draft,
        ))

        connection.commit()

        print(
            f"Response draft {decision.lower()} successfully!"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# =========================================
# STAFF REVIEW HISTORY
# =========================================

def get_review_history(case_id):
    """Retrieve all staff review decisions for a case."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM review_history
            WHERE case_id = ?
            ORDER BY reviewed_at DESC
        """, (case_id,))

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:
        connection.close()


# =========================================
# INITIALIZE WHEN RUN DIRECTLY
# =========================================

if __name__ == "__main__":
    init_db()
