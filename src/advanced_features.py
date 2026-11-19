"""Advanced intelligence features for RailResolve AI.

This module adds six department-facing capabilities without changing the
existing complaint, RAG, response-draft, or staff-review tables:
1. Explainable complaint-priority recommendations
2. Potential duplicate complaint detection
3. Configurable demo SLA tracking and escalation flags
4. Suggested department routing and staff assignment
5. Reported-location hotspot summaries
6. Database-driven management insights

Important: Priority, routing, and SLA values are prototype recommendations.
They are not official Indian Railways rules or service-level commitments.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # The existing project audit already checks scikit-learn.
    TfidfVectorizer = None
    cosine_similarity = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "railresolve.db"

DEFAULT_SLA_TARGETS = {
    "High": 24,
    "Medium": 72,
    "Low": 120,
}

DEPARTMENT_MAP = {
    "refund": "Commercial / Refunds (suggested)",
    "ticketing": "Commercial / Ticketing (suggested)",
    "delay": "Operations / Train Running (suggested)",
    "cleanliness": "Station Management (suggested)",
    "facilities": "Station Management (suggested)",
    "lost property": "Lost Property / Concerned Station (suggested)",
    "staff assistance": "Passenger Services (suggested)",
    "information": "Passenger Services / Information (suggested)",
}

CUSTOM_DEPARTMENTS = [
    "Commercial / Refunds",
    "Commercial / Ticketing",
    "Operations / Train Running",
    "Station Management",
    "Lost Property / Concerned Station",
    "Passenger Services",
    "Other / Manual Routing",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime | None = None) -> str:
    return (value or _utc_now()).astimezone(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    """Open the same project-local database used by the existing application."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_advanced_schema() -> None:
    """Create extension tables only; existing application tables are untouched."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS case_enrichment (
                case_id TEXT PRIMARY KEY,
                suggested_priority TEXT NOT NULL,
                priority_score INTEGER NOT NULL DEFAULT 0,
                priority_reasons TEXT,
                confirmed_priority TEXT,
                suggested_department TEXT,
                assigned_department TEXT,
                assigned_to TEXT,
                sla_target_hours INTEGER,
                sla_due_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS duplicate_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                possible_duplicate_case_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                reason TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Pending',
                reviewed_by TEXT,
                review_notes TEXT,
                first_detected_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(case_id, possible_duplicate_case_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS advanced_feature_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        now = _utc_iso()
        for priority, hours in DEFAULT_SLA_TARGETS.items():
            key = f"sla_{priority.lower()}_hours"
            connection.execute(
                """
                INSERT OR IGNORE INTO advanced_feature_settings
                (setting_key, setting_value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, str(hours), now),
            )
        connection.commit()


def _safe_json(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _case_text(case: dict[str, Any]) -> str:
    parts = [
        str(case.get("complaint_text") or ""),
        str(case.get("category") or ""),
        str(case.get("train_number") or ""),
        str(case.get("journey_date") or ""),
        str(case.get("location") or ""),
    ]
    return " ".join(part for part in parts if part).strip()


def _structured_details(case: dict[str, Any]) -> dict[str, Any]:
    detected = _safe_json(case.get("detected_information"), {})
    if not isinstance(detected, dict):
        return {}
    details = detected.get("structured_details", {})
    return details if isinstance(details, dict) else {}


def _case_train_number(case: dict[str, Any]) -> str:
    details = _structured_details(case)
    return str(case.get("train_number") or details.get("train_number") or "").strip()


def _case_journey_date(case: dict[str, Any]) -> str:
    details = _structured_details(case)
    return str(case.get("journey_date") or details.get("journey_date") or "").strip()


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            text = str(value).strip().replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        # Existing project timestamps may be naive local-time strings. Let the
        # operating system's configured local timezone interpret them before
        # converting to UTC; do not silently treat local time as UTC.
        return parsed.astimezone().astimezone(timezone.utc)
    return parsed.astimezone(timezone.utc)


def recommend_priority(case: dict[str, Any]) -> dict[str, Any]:
    """Return an explainable rules-based priority suggestion."""
    text = _case_text(case).lower()
    score = 0
    reasons: list[str] = []

    critical_terms = (
        "accident", "derailment", "fire", "smoke", "injury", "medical emergency",
        "medical help", "violence", "assault", "unsafe", "security threat",
        "stranded", "trapped", "emergency", "life-threatening",
    )
    urgent_terms = (
        "urgent", "immediately", "immediate assistance", "elderly passenger",
        "senior citizen alone", "child alone", "pregnant passenger", "disabled passenger",
        "cannot reach assistance", "no one is helping", "dangerous condition",
    )
    impact_terms = (
        "several hours", "many passengers", "multiple passengers", "no drinking water",
        "toilet not working", "cancelled", "canceled", "repeat complaint", "repeated",
        "refund pending", "not received", "lost my bag", "lost luggage",
    )

    matched_critical = [term for term in critical_terms if term in text]
    matched_urgent = [term for term in urgent_terms if term in text]
    matched_impact = [term for term in impact_terms if term in text]

    if matched_critical:
        score += 6
        reasons.append("Potential safety, medical, or emergency wording detected")
    if matched_urgent:
        score += 3
        reasons.append("Urgent passenger-assistance wording detected")
    if matched_impact:
        score += 2
        reasons.append("Potentially significant passenger-impact wording detected")

    category = str(case.get("category") or "").strip().lower()
    if category == "delay" and ("several hours" in text or "hours" in text):
        score += 1
        reasons.append("Extended train-delay wording detected")

    if score >= 6:
        priority = "High"
    elif score >= 2:
        priority = "Medium"
    else:
        priority = "Low"

    if not reasons:
        reasons.append("No configured urgent-risk phrases detected; staff should verify priority")

    return {"priority": priority, "score": score, "reasons": reasons}


def suggest_department(case: dict[str, Any]) -> str:
    category = str(case.get("category") or "").strip().lower()
    return DEPARTMENT_MAP.get(category, "Passenger Services / Manual Review (suggested)")


def get_sla_targets() -> dict[str, int]:
    initialize_advanced_schema()
    targets = dict(DEFAULT_SLA_TARGETS)
    with _connect() as connection:
        rows = connection.execute(
            "SELECT setting_key, setting_value FROM advanced_feature_settings"
        ).fetchall()
    for row in rows:
        key = str(row["setting_key"])
        if key.startswith("sla_") and key.endswith("_hours"):
            priority = key[len("sla_") : -len("_hours")].title()
            if priority in targets:
                try:
                    targets[priority] = max(1, int(row["setting_value"]))
                except (TypeError, ValueError):
                    pass
    return targets


def save_sla_targets(targets: dict[str, int]) -> None:
    initialize_advanced_schema()
    now = _utc_iso()
    with _connect() as connection:
        for priority in ("High", "Medium", "Low"):
            hours = int(targets[priority])
            if hours < 1 or hours > 24 * 90:
                raise ValueError("SLA demo targets must be between 1 and 2160 hours.")
            connection.execute(
                """
                INSERT INTO advanced_feature_settings (setting_key, setting_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET
                    setting_value=excluded.setting_value,
                    updated_at=excluded.updated_at
                """,
                (f"sla_{priority.lower()}_hours", str(hours), now),
            )
        connection.commit()


def sync_case_enrichment(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Refresh recommendations while preserving staff-confirmed overrides."""
    initialize_advanced_schema()
    targets = get_sla_targets()
    now = _utc_iso()
    output: dict[str, dict[str, Any]] = {}

    with _connect() as connection:
        for case in cases:
            case_id = str(case.get("case_id") or "").strip()
            if not case_id:
                continue
            priority_result = recommend_priority(case)
            department = suggest_department(case)
            existing = connection.execute(
                "SELECT * FROM case_enrichment WHERE case_id = ?", (case_id,)
            ).fetchone()

            confirmed_priority = existing["confirmed_priority"] if existing else None
            assigned_department = existing["assigned_department"] if existing else None
            assigned_to = existing["assigned_to"] if existing else None
            effective_priority = confirmed_priority or priority_result["priority"]
            target_hours = targets.get(effective_priority, targets["Medium"])
            created = _parse_datetime(case.get("created_at")) or _utc_now()
            due_at = created + timedelta(hours=target_hours)

            connection.execute(
                """
                INSERT INTO case_enrichment (
                    case_id, suggested_priority, priority_score, priority_reasons,
                    confirmed_priority, suggested_department, assigned_department,
                    assigned_to, sla_target_hours, sla_due_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    suggested_priority=excluded.suggested_priority,
                    priority_score=excluded.priority_score,
                    priority_reasons=excluded.priority_reasons,
                    suggested_department=excluded.suggested_department,
                    sla_target_hours=excluded.sla_target_hours,
                    sla_due_at=excluded.sla_due_at,
                    updated_at=excluded.updated_at
                """,
                (
                    case_id,
                    priority_result["priority"],
                    priority_result["score"],
                    json.dumps(priority_result["reasons"]),
                    confirmed_priority,
                    department,
                    assigned_department,
                    assigned_to,
                    target_hours,
                    due_at.isoformat(timespec="seconds"),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM case_enrichment WHERE case_id = ?", (case_id,)
            ).fetchone()
            output[case_id] = dict(row)
        connection.commit()
    return output


def save_case_priority(case_id: str, priority: str) -> None:
    if priority not in ("High", "Medium", "Low"):
        raise ValueError("Priority must be High, Medium, or Low.")
    initialize_advanced_schema()
    targets = get_sla_targets()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM case_enrichment WHERE case_id = ?", (case_id,)
        ).fetchone()
        if not row:
            raise ValueError("Case intelligence has not been initialized for this case.")
        case_row = connection.execute(
            "SELECT created_at FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        created = _parse_datetime(case_row["created_at"] if case_row else None) or _utc_now()
        due_at = created + timedelta(hours=targets[priority])
        connection.execute(
            """
            UPDATE case_enrichment
            SET confirmed_priority = ?, sla_target_hours = ?, sla_due_at = ?, updated_at = ?
            WHERE case_id = ?
            """,
            (priority, targets[priority], due_at.isoformat(timespec="seconds"), _utc_iso(), case_id),
        )
        connection.commit()


def save_case_assignment(case_id: str, department: str, assigned_to: str) -> None:
    initialize_advanced_schema()
    if not department.strip():
        raise ValueError("Please select a department.")
    with _connect() as connection:
        changed = connection.execute(
            """
            UPDATE case_enrichment
            SET assigned_department = ?, assigned_to = ?, updated_at = ?
            WHERE case_id = ?
            """,
            (department.strip(), assigned_to.strip() or None, _utc_iso(), case_id),
        ).rowcount
        if not changed:
            raise ValueError("Case intelligence has not been initialized for this case.")
        connection.commit()


def detect_duplicate_candidates(
    cases: list[dict[str, Any]], threshold: float = 0.55, max_cases: int = 500
) -> list[dict[str, Any]]:
    """Find likely duplicate pairs; never merges, deletes, or auto-closes cases."""
    initialize_advanced_schema()
    recent = [case for case in cases if case.get("case_id") and _case_text(case)]
    recent = recent[:max_cases]
    if len(recent) < 2 or TfidfVectorizer is None or cosine_similarity is None:
        return list_duplicate_candidates()

    texts = [_case_text(case) for case in recent]
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(texts)
        similarities = cosine_similarity(matrix)
    except ValueError:
        return list_duplicate_candidates()

    now = _utc_iso()
    with _connect() as connection:
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                similarity = float(similarities[i, j])
                left, right = recent[i], recent[j]
                same_train_date = (
                    _case_train_number(left)
                    and _case_train_number(left) == _case_train_number(right)
                    and _case_journey_date(left)
                    and _case_journey_date(left) == _case_journey_date(right)
                )
                same_category = str(left.get("category", "")).lower() == str(right.get("category", "")).lower()
                contextual_match = bool(same_train_date and same_category)
                if similarity < threshold and not (contextual_match and similarity >= 0.18):
                    continue

                left_id, right_id = sorted((str(left["case_id"]), str(right["case_id"])))
                reasons = [f"Text similarity: {similarity:.0%}"]
                if contextual_match:
                    reasons.append("Same train number, journey date, and complaint category")
                reason = "; ".join(reasons)
                connection.execute(
                    """
                    INSERT INTO duplicate_candidates (
                        case_id, possible_duplicate_case_id, similarity, reason,
                        review_status, first_detected_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, 'Pending', ?, ?)
                    ON CONFLICT(case_id, possible_duplicate_case_id) DO UPDATE SET
                        similarity=excluded.similarity,
                        reason=excluded.reason,
                        last_seen_at=excluded.last_seen_at
                    """,
                    (left_id, right_id, similarity, reason, now, now),
                )
        connection.commit()
    return list_duplicate_candidates()


def list_duplicate_candidates() -> list[dict[str, Any]]:
    initialize_advanced_schema()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM duplicate_candidates
            ORDER BY CASE review_status WHEN 'Pending' THEN 0 ELSE 1 END,
                     similarity DESC, last_seen_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def review_duplicate_candidate(
    candidate_id: int, decision: str, reviewer: str, notes: str
) -> None:
    allowed = {"Pending", "Confirmed Related", "Not Duplicate", "Needs Investigation"}
    if decision not in allowed:
        raise ValueError("Invalid duplicate-review decision.")
    if decision in {"Not Duplicate", "Needs Investigation"} and not notes.strip():
        raise ValueError("Please add review notes for this decision.")
    with _connect() as connection:
        changed = connection.execute(
            """
            UPDATE duplicate_candidates
            SET review_status = ?, reviewed_by = ?, review_notes = ?
            WHERE id = ?
            """,
            (decision, reviewer.strip() or None, notes.strip() or None, int(candidate_id)),
        ).rowcount
        if not changed:
            raise ValueError("Duplicate candidate was not found.")
        connection.commit()


def calculate_sla(case: dict[str, Any], enrichment: dict[str, Any]) -> dict[str, Any]:
    status = str(case.get("status") or "").strip().lower()
    if status in {"resolved", "closed"}:
        return {
            "case_id": case.get("case_id"),
            "sla_status": "Stopped – case closed/resolved",
            "hours_remaining": None,
            "due_at": enrichment.get("sla_due_at"),
            "age_hours": None,
        }

    created = _parse_datetime(case.get("created_at"))
    due = _parse_datetime(enrichment.get("sla_due_at"))
    now = _utc_now()
    age_hours = round((now - created).total_seconds() / 3600, 1) if created else None
    if due is None:
        return {
            "case_id": case.get("case_id"),
            "sla_status": "Target unavailable",
            "hours_remaining": None,
            "due_at": None,
            "age_hours": age_hours,
        }
    remaining = (due - now).total_seconds() / 3600
    return {
        "case_id": case.get("case_id"),
        "sla_status": "Overdue" if remaining < 0 else "On Track",
        "hours_remaining": round(remaining, 1),
        "due_at": due.isoformat(timespec="minutes"),
        "age_hours": age_hours,
    }


def extract_reported_location(case: dict[str, Any]) -> tuple[str, str]:
    """Extract a station-like phrase from complaint text, otherwise use stored location."""
    text = str(case.get("complaint_text") or "")
    patterns = (
        r"\bat\s+([A-Z][A-Za-z .'-]{1,55}?(?:Junction|Station|Terminal|Central))(?=,|\.|;|\s+platform\b|$)",
        r"\b(?:station|junction)\s+(?:at\s+)?([A-Z][A-Za-z .'-]{1,45})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" ,.-")
            if value:
                return value, "Place phrase extracted from complaint text"

    details = _structured_details(case)
    location = str(case.get("location") or details.get("location") or "").strip()
    if location:
        return location, "Stored passenger location detail"
    return "Not provided", "No location available"


def build_hotspot_summary(cases: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for case in cases:
        place, source = extract_reported_location(case)
        if place == "Not provided":
            continue
        rows.append({
            "Reported Location": place,
            "Category": str(case.get("category") or "Uncategorised"),
            "Case ID": str(case.get("case_id") or ""),
            "Location Source": source,
        })
    if not rows:
        return pd.DataFrame(columns=["Reported Location", "Complaints", "Top Category"])
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("Reported Location")
        .agg(
            Complaints=("Case ID", "count"),
            **{"Top Category": ("Category", lambda values: values.value_counts().index[0])},
        )
        .reset_index()
        .sort_values("Complaints", ascending=False)
    )
    return summary


def build_management_insights(
    cases: list[dict[str, Any]],
    enrichments: dict[str, dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
) -> list[str]:
    if not cases:
        return ["No complaint records are available yet. Submit sample cases to populate management insights."]

    frame = pd.DataFrame(cases)
    total = len(frame)
    category_counts = frame.get("category", pd.Series(dtype=str)).fillna("Uncategorised").value_counts()
    status_series = frame.get("status", pd.Series(dtype=str)).fillna("Unknown").astype(str)
    active_mask = ~status_series.str.lower().isin({"resolved", "closed"})
    active = int(active_mask.sum())
    pending_review = int(
        frame.get("staff_review_status", pd.Series("Pending", index=frame.index))
        .fillna("Pending").astype(str).str.lower().eq("pending").sum()
    )
    high_priority = sum(
        1 for data in enrichments.values()
        if (data.get("confirmed_priority") or data.get("suggested_priority")) == "High"
    )
    pending_duplicates = sum(1 for row in duplicate_rows if row.get("review_status") == "Pending")

    insights = [f"{total} complaint cases are currently stored in the prototype database."]
    if not category_counts.empty:
        top_category = str(category_counts.index[0])
        top_count = int(category_counts.iloc[0])
        insights.append(f"{top_category} is the most common recorded category ({top_count} cases, {top_count / total:.1%} of all cases).")
    insights.append(f"{active} cases are not marked Resolved or Closed ({active / total:.1%} of recorded cases).")
    insights.append(f"{pending_review} cases currently show a Pending staff-review status.")
    insights.append(f"{high_priority} cases have a High priority recommendation or staff-confirmed priority.")
    insights.append(f"{pending_duplicates} potential duplicate pairs are awaiting review.")
    insights.append("These insights describe only the records in this prototype database; they are not live railway-wide statistics.")
    return insights
