from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ContractError


EVENT_CATEGORIES = (
    "macro",
    "regulation",
    "exchange",
    "stablecoin",
    "institutional_flow",
    "onchain_stress",
    "derivatives_stress",
    "security",
    "other",
)
REQUIRED_COLUMNS = (
    "event_id",
    "published_at_utc",
    "available_at_utc",
    "source",
    "source_url",
    "category",
    "severity",
    "headline",
    "body",
    "content_sha256",
)


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ContractError("event timestamps must include UTC offset")
    return parsed.astimezone(timezone.utc)


def normalized_content(headline: str, body: str) -> str:
    return " ".join((str(headline).strip() + "\n" + str(body).strip()).split())


def content_sha256(headline: str, body: str) -> str:
    return hashlib.sha256(normalized_content(headline, body).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    published_at: datetime
    available_at: datetime
    source: str
    source_url: str
    category: str
    severity: int
    headline: str
    body: str
    content_sha256: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "EventRecord":
        if set(row) != set(REQUIRED_COLUMNS):
            raise ContractError("event-ledger columns changed")
        event_id = row["event_id"].strip()
        source = row["source"].strip()
        category = row["category"].strip()
        headline = row["headline"].strip()
        body = row["body"].strip()
        if not event_id or not source or not headline or not body:
            raise ContractError("event identity, source, headline, and body are required")
        if category not in EVENT_CATEGORIES:
            raise ContractError(f"unsupported event category: {category}")
        try:
            severity = int(row["severity"])
        except ValueError as exc:
            raise ContractError("severity must be an integer") from exc
        if severity not in (1, 2, 3, 4, 5):
            raise ContractError("severity must be between 1 and 5")
        published = parse_utc(row["published_at_utc"])
        available = parse_utc(row["available_at_utc"])
        if available < published:
            raise ContractError("available_at cannot precede published_at")
        digest = content_sha256(headline, body)
        if row["content_sha256"].strip().lower() != digest:
            raise ContractError(f"content hash mismatch for {event_id}")
        return cls(
            event_id,
            published,
            available,
            source,
            row["source_url"].strip(),
            category,
            severity,
            headline,
            body,
            digest,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "published_at_utc": self.published_at.isoformat(),
            "available_at_utc": self.available_at.isoformat(),
            "source": self.source,
            "category": self.category,
            "severity": self.severity,
            "headline": self.headline,
            "body": self.body,
            "content_sha256": self.content_sha256,
        }


def load_event_ledger(path: Path) -> list[EventRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ContractError("event-ledger header does not match the frozen schema")
        records = [EventRecord.from_row(dict(row)) for row in reader]
    if not records:
        raise ContractError("event ledger is empty")
    identifiers = [record.event_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ContractError("event identifiers are not unique")
    return sorted(records, key=lambda item: (item.available_at, item.event_id))


def validate_manifest(path: Path, ledger_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_id",
        "event_csv_sha256",
        "collection_method",
        "revision_policy",
        "raw_text_preserved",
        "point_in_time_availability_verified",
        "coverage_start_utc",
        "coverage_end_utc",
    }
    if set(value) != required:
        raise ContractError("event manifest keys changed")
    if value["event_csv_sha256"] != file_sha256(ledger_path):
        raise ContractError("event ledger hash does not match its manifest")
    if value["raw_text_preserved"] is not True:
        raise ContractError("raw event text was not preserved")
    if value["point_in_time_availability_verified"] is not True:
        raise ContractError("point-in-time availability was not verified")
    if not str(value["collection_method"]).strip() or not str(value["revision_policy"]).strip():
        raise ContractError("event collection and revision policies are required")
    parse_utc(value["coverage_start_utc"])
    parse_utc(value["coverage_end_utc"])
    return value


def events_available_asof(records: list[EventRecord], decision_time_utc: str) -> list[EventRecord]:
    cutoff = parse_utc(decision_time_utc)
    return [record for record in records if record.available_at <= cutoff]

