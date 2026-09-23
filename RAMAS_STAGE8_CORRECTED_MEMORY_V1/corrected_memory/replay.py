"""Read-only replay of an archived arm's durable journal.

Used to prove that this package's engine reproduces a frozen Stage 6.4 arm
byte-for-byte before any GPU time is spent.  It opens archived call records for
reading only, never writes into an existing artifact directory, and never
contacts a model: if a payload differs from the archived one by a single byte,
its digest changes, the record cannot be matched, and the replay stops.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stage63lib.journal import digest


class ReplayMismatch(RuntimeError):
    pass


class ReplayJournal:
    """Journal-compatible reader over an archived ``calls/`` directory."""

    def __init__(self, archive_arm_dir: Path, *, strict_payload: bool = True):
        self.calls = Path(archive_arm_dir).resolve() / "calls"
        if not self.calls.is_dir():
            raise ReplayMismatch(f"Archived call directory is missing: {self.calls}")
        self.strict_payload = strict_payload
        self.new_calls = 0
        self.reused_calls = 0
        self.matched = 0
        self.first_mismatch: dict[str, Any] | None = None

    def complete(self, key: str, payload: dict[str, Any]):
        path = self.calls / (key + ".json")
        if not path.is_file():
            raise ReplayMismatch(f"No archived response for {key}; replay cannot invent one")
        entry = json.loads(path.read_text(encoding="utf-8"))
        request_hash = digest(payload)
        if entry["request_sha256"] != request_hash:
            if self.first_mismatch is None:
                self.first_mismatch = {
                    "key": key,
                    "archived_request_sha256": entry["request_sha256"],
                    "rebuilt_request_sha256": request_hash,
                    "differing_top_level_fields": sorted(
                        k for k in set(entry["request"]) | set(payload)
                        if entry["request"].get(k) != payload.get(k)
                    ),
                }
            raise ReplayMismatch(
                f"Rebuilt payload differs from the archived request at {key}. "
                f"archived={entry['request_sha256'][:12]} rebuilt={request_hash[:12]}"
            )
        if self.strict_payload and entry["request"] != payload:
            raise ReplayMismatch(f"Payload digest matched but content differs at {key}")
        self.matched += 1
        self.reused_calls += 1
        return entry["raw_response"], entry["provider_response"], entry["latency_seconds"]


class RefusingProvider:
    """Placeholder provider that makes replay attempts to call a model fail loudly."""

    kind = "replay_only"
    pinned = {"kind": "replay_only", "model_calls": 0}

    def complete(self, payload):
        raise ReplayMismatch("Replay mode must never contact a model")

    def assert_identity(self):
        return None
