"""Small, dependency-free helpers shared by the runtime and the tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


# ----------------------------------------------------------------------
# time / identity
# ----------------------------------------------------------------------
def now_iso() -> str:
    """UTC timestamp with microsecond precision so ordering is unambiguous."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def short_id(prefix: str = "", length: int = 10) -> str:
    raw = f"{prefix}{time.time_ns()}{os.getpid()}"
    d = hashlib.sha1(raw.encode()).hexdigest()[:length]
    return f"{prefix}{d}" if prefix else d


def digest(*parts: Any, length: int = 16) -> str:
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def slugify(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not slug:
        slug = "objective"
    return slug[:max_length].rstrip("-")


# ----------------------------------------------------------------------
# filesystem
# ----------------------------------------------------------------------
def atomic_write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_write_json(path: Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def append_jsonl(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:3.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def safe_join(root: Path, *parts: str) -> Path:
    """Join *parts* onto *root*, refusing to escape the root directory."""
    root = Path(root).resolve()
    candidate = Path(os.path.expanduser(str(Path(root, *parts))))
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes workspace: {resolved}")
    return resolved



# ----------------------------------------------------------------------
# process execution
# ----------------------------------------------------------------------
@dataclass
class ProcResult:
    code: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    command: Sequence[str] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def output(self) -> str:
        joined = self.stdout
        if self.stderr:
            joined = f"{joined}\n{self.stderr}" if joined else self.stderr
        return joined.strip()

    def tail(self, limit: int = 2000) -> str:
        text = self.output
        return text if len(text) <= limit else "..." + text[-limit:]

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "stdout": self.stdout[-8000:],
            "stderr": self.stderr[-8000:],
            "duration": round(self.duration, 3),
            "command": list(self.command),
        }


def run_cmd(
    command: Sequence[str],
    cwd: Path | str | None = None,
    timeout: float = 120.0,
    env: dict | None = None,
    stdin_text: str | None = None,
) -> ProcResult:
    started = time.time()
    merged_env = {**os.environ, **(env or {})}
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
        )
        return ProcResult(
            code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration=time.time() - started,
            command=list(command),
        )
    except FileNotFoundError:
        return ProcResult(
            code=127,
            stderr=f"command not found: {command[0]}",
            duration=time.time() - started,
            command=list(command),
        )
    except subprocess.TimeoutExpired:
        return ProcResult(
            code=124,
            stderr=f"timed out after {timeout}s: {' '.join(str(c) for c in command)}",
            duration=time.time() - started,
            command=list(command),
        )
    except OSError as exc:  # pragma: no cover - platform specific
        return ProcResult(
            code=126,
            stderr=f"{type(exc).__name__}: {exc}",
            duration=time.time() - started,
            command=list(command),
        )


def which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def ffmpeg_bin() -> str | None:
    """Locate an ffmpeg binary: PATH first, then the pip-bundled one."""
    found = which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:  # pragma: no cover - optional dependency
        return None
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def chunked(items: Sequence[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, size)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "it", "its", "i", "we",
    "you", "they", "he", "she", "me", "my", "our", "your", "their", "can",
    "will", "would", "should", "could", "do", "does", "did", "not", "no",
    "make", "made", "create", "build", "want", "need", "please", "using",
    "use", "about", "into", "also", "some", "any", "all", "so", "up", "out",
}


def keywords(text: str, limit: int = 12) -> list[str]:
    seen: list[str] = []
    for token in tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    return seen
