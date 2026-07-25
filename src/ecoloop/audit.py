from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

def append_event(path: str | Path, event: dict) -> None:
    def default(value):
        if isinstance(value, datetime): return value.isoformat()
        if is_dataclass(value): return asdict(value)
        raise TypeError(f"not JSON serializable: {type(value)!r}")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, default=default, sort_keys=True) + "\n")
