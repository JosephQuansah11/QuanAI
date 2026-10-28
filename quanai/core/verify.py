"""Verification: turn requirements into executed, observable checks.

This module is the "constant check" half of the agent.  Nothing is trusted
because a tool said it succeeded - a requirement is only marked verified when
a check has actually opened the produced file and measured it.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..errors import CheckError
from ..util import run_cmd, safe_join, ffmpeg_bin
from .task import Check, Evidence, Objective, Requirement, Status, Task

CheckFn = Callable[..., tuple[bool, str, Any]]
_REGISTRY: dict[str, CheckFn] = {}


def register_check(kind: str) -> Callable[[CheckFn], CheckFn]:
    def wrapper(fn: CheckFn) -> CheckFn:
        _REGISTRY[kind] = fn
        return fn

    return wrapper


def known_checks() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class VerifyContext:
    workspace: Path
    objective: Objective | None = None
    task: Task | None = None

    def resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return safe_join(self.workspace, path)


# ----------------------------------------------------------------------
# generic file checks
# ----------------------------------------------------------------------
@register_check("file_exists")
def _file_exists(ctx: VerifyContext, path: str, allow_glob: bool = True, **_: Any):
    if allow_glob and any(ch in path for ch in "*?["):
        matches = sorted(ctx.workspace.glob(path))
        if not matches:
            return False, f"no files match {path!r}", []
        return True, f"{len(matches)} file(s) match {path!r}", [str(m) for m in matches]
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    if target.is_dir():
        return False, f"expected a file but found a directory: {path}", str(target)
    return True, f"exists ({target.stat().st_size} bytes)", str(target)


@register_check("file_min_bytes")
def _file_min_bytes(ctx: VerifyContext, path: str, min_bytes: int = 1, **_: Any):
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    size = target.stat().st_size
    ok = size >= int(min_bytes)
    return ok, f"{path} is {size} bytes (required >= {min_bytes})", {"bytes": size}


@register_check("text_contains")
def _text_contains(ctx: VerifyContext, path: str, needles: Any = None, min_chars: int = 1, **_: Any):
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    text = target.read_text(encoding="utf-8", errors="replace")
    needs = needles if isinstance(needles, list) else ([needles] if needles else [])
    lowered = text.lower()
    missing = [n for n in needs if str(n).lower() not in lowered]
    if missing:
        return False, f"{path} is missing: {', '.join(map(str, missing))}", {"chars": len(text)}
    if len(text) < int(min_chars):
        return False, f"{path} has only {len(text)} characters (required {min_chars})", len(text)
    return True, f"{path} contains all {len(needs)} required phrase(s), {len(text)} chars", len(text)


@register_check("no_placeholders")
def _no_placeholders(ctx: VerifyContext, path: str, max_hits: int = 0, **_: Any):
    """Reject stub code and filler copy, in a single file or a whole project."""
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    pattern = re.compile(
        r"\b(TODO|FIXME|lorem ipsum|NotImplementedError|Coming soon|placeholder text)\b", re.I
    )
    files = [target] if target.is_file() else [p for p in sorted(target.rglob("*")) if p.is_file()]
    readable = {".py", ".js", ".ts", ".html", ".css", ".md", ".txt", ".json", ".yaml", ".yml", ".dart"}
    hits: list[str] = []
    scanned = 0
    for file in files:
        if file.suffix.lower() not in readable or file.stat().st_size > 512_000:
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        for match in pattern.findall(text):
            hits.append(f"{file.name}: {match}")
    ok = len(hits) <= int(max_hits)
    return ok, f"{len(hits)} placeholder marker(s) across {scanned} file(s) in {path}", hits[:10]


@register_check("json_valid")
def _json_valid(ctx: VerifyContext, path: str, keys: Any = None, **_: Any):
    target = ctx.resolve(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, f"missing file: {path}", str(target)
    except json.JSONDecodeError as exc:
        return False, f"{path} is not valid JSON: {exc}", str(target)
    if isinstance(keys, str):
        keys = [keys]
    missing = [k for k in (keys or []) if not isinstance(data, dict) or k not in data]
    if missing:
        return False, f"{path} missing keys: {', '.join(missing)}", list(data)[:10]
    return True, f"{path} is valid JSON", type(data).__name__


@register_check("zip_valid")
def _zip_valid(ctx: VerifyContext, path: str, required: Any = None, **_: Any):
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    try:
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            bad = archive.testzip()
    except zipfile.BadZipFile as exc:
        return False, f"{path} is not a valid zip container: {exc}", str(target)
    if bad:
        return False, f"corrupt member in {path}: {bad}", bad
    if isinstance(required, str):
        required = [required]
    missing = [r for r in (required or []) if not any(name.startswith(r) for name in names)]
    if missing:
        return False, f"{path} missing members: {', '.join(missing)}", len(names)
    return True, f"{path} is a valid container with {len(names)} members", len(names)



# ----------------------------------------------------------------------
# artifact checks
# ----------------------------------------------------------------------
@register_check("pptx_valid")
def _pptx_valid(
    ctx: VerifyContext,
    path: str,
    min_slides: int = 1,
    required_terms: Any = None,
    min_chars: int = 0,
    **_: Any,
):
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    try:
        from pptx import Presentation  # type: ignore
    except ImportError:  # pragma: no cover - declared dependency
        return False, "python-pptx is not installed, cannot verify deck", None
    try:
        deck = Presentation(str(target))
        slides = list(deck.slides)
        texts = []
        for slide in slides:
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                    texts.append(shape.text_frame.text)
                if getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        texts.extend(cell.text for cell in row.cells)
        body = "\n".join(texts)
    except Exception as exc:
        return False, f"cannot open deck {path}: {type(exc).__name__}: {exc}", str(target)

    problems = []
    if len(slides) < int(min_slides):
        problems.append(f"{len(slides)} slide(s) < required {min_slides}")
    if len(body) < int(min_chars):
        problems.append(f"{len(body)} characters of text < required {min_chars}")
    if isinstance(required_terms, str):
        required_terms = [required_terms]
    missing = [t for t in (required_terms or []) if str(t).lower() not in body.lower()]
    if missing:
        problems.append("deck never mentions: " + ", ".join(map(str, missing)))
    observed = {
        "slides": len(slides),
        "characters": len(body),
        "slide_titles": [s.splitlines()[0][:80] for s in texts[:20] if s.strip()][:12],
    }
    if problems:
        return False, "; ".join(problems), observed
    return True, f"deck has {len(slides)} slides and {len(body)} characters of text", observed


@register_check("image_valid")
def _image_valid(
    ctx: VerifyContext,
    path: str,
    min_width: int = 32,
    min_height: int = 32,
    min_variance: float = 20.0,
    **_: Any,
):
    from ..probe import image_info

    target = ctx.resolve(path)
    info = image_info(target)
    if not info.exists:
        return False, f"missing file: {path}", str(target)
    if info.error:
        return False, f"{path} is not a readable image: {info.error}", info.to_dict()
    if info.width < int(min_width) or info.height < int(min_height):
        return False, f"{path} is {info.width}x{info.height}, required >= {min_width}x{min_height}", info.to_dict()
    variance = 0.0
    try:
        import numpy as np
        from PIL import Image  # type: ignore

        with Image.open(target) as img:
            arr = np.asarray(img.convert("L"), dtype="float32")
        # keep native detail (resizing would average noise away) but bound cost
        stride = max(1, arr.shape[0] // 256, arr.shape[1] // 256)
        variance = float(arr[::stride, ::stride].std())
    except Exception:
        variance = float("nan")
    if variance == variance and variance < float(min_variance):
        return False, f"{path} looks blank (std {variance:.1f} < {min_variance})", info.to_dict()
    return True, f"{info.width}x{info.height} {info.format} image, detail std {variance:.1f}", info.to_dict()


@register_check("video_valid")
def _video_valid(
    ctx: VerifyContext,
    path: str,
    min_seconds: float = 1.0,
    min_width: int = 16,
    require_audio: bool = False,
    **_: Any,
):
    from ..probe import probe

    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    info = probe(target)
    if info.error:
        return False, f"cannot read {path}: {info.error}", info.to_dict()
    if not info.has_video:
        return False, f"{path} has no video stream", info.to_dict()
    if info.duration < float(min_seconds):
        return False, f"{path} runs {info.duration}s, required >= {min_seconds}s", info.to_dict()
    if info.width and info.width < int(min_width):
        return False, f"{path} width {info.width} < {min_width}", info.to_dict()
    if require_audio and not info.has_audio:
        return False, f"{path} has no audio track", info.to_dict()
    return True, (
        f"{info.width}x{info.height} {info.video_codec} video, "
        f"{info.duration}s, audio={info.has_audio}"
    ), info.to_dict()


@register_check("audio_valid")
def _audio_valid(ctx: VerifyContext, path: str, min_seconds: float = 0.5, **_: Any):
    from ..probe import probe

    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    info = probe(target)
    if info.error:
        return False, f"cannot read {path}: {info.error}", info.to_dict()
    if not info.has_audio:
        return False, f"{path} has no audio stream", info.to_dict()
    if info.duration < float(min_seconds):
        return False, f"{path} runs {info.duration}s, required >= {min_seconds}s", info.to_dict()
    return True, f"{info.duration}s {info.audio_codec} audio", info.to_dict()



# ----------------------------------------------------------------------
# source-code / project checks
# ----------------------------------------------------------------------
@register_check("python_compiles")
def _python_compiles(ctx: VerifyContext, path: str, **_: Any):
    if any(ch in path for ch in "*?["):
        targets = sorted(ctx.workspace.glob(path))
    else:
        resolved = ctx.resolve(path)
        targets = [resolved] if resolved.is_file() else sorted(resolved.rglob("*.py"))
    targets = [t for t in targets if t.suffix == ".py"]
    if not targets:
        return False, f"no python files found for {path!r}", []
    broken = []
    for target in targets:
        try:
            source = target.read_text(encoding="utf-8", errors="replace")
            compile(source, str(target), "exec")
        except SyntaxError as exc:
            broken.append(f"{target.name}:{exc.lineno}: {exc.msg}")
    if broken:
        return False, "syntax errors: " + "; ".join(broken[:5]), broken
    return True, f"{len(targets)} python file(s) compile cleanly", [str(t) for t in targets]


@register_check("js_syntax")
def _js_syntax(ctx: VerifyContext, path: str, **_: Any):
    import shutil as _shutil

    node = _shutil.which("node")
    if not node:
        return False, "node is not installed, cannot syntax-check javascript", None
    resolved = ctx.resolve(path)
    targets = [resolved] if resolved.is_file() else sorted(resolved.rglob("*.js"))
    if not targets:
        return False, f"no javascript files found for {path!r}", []
    broken = []
    for target in targets:
        result = run_cmd([node, "--check", str(target)], timeout=40)
        if not result.ok:
            broken.append(f"{target.name}: {result.tail(300)}")
    if broken:
        return False, "javascript errors: " + " | ".join(broken[:3]), broken
    return True, f"{len(targets)} javascript file(s) parse cleanly", [str(t) for t in targets]


@register_check("scaffold_files")
def _scaffold_files(
    ctx: VerifyContext,
    path: str,
    required: Any = None,
    any_of: Any = None,
    min_bytes: int = 24,
    **_: Any,
):
    root = ctx.resolve(path)
    if not root.exists():
        return False, f"missing directory: {path}", str(root)
    if isinstance(required, str):
        required = [required]
    if isinstance(any_of, str):
        any_of = [any_of]
    missing, empty, present = [], [], []
    for rel in required or []:
        target = root / rel
        if not target.exists():
            missing.append(rel)
        elif target.is_file() and target.stat().st_size < int(min_bytes):
            empty.append(rel)
        else:
            present.append(rel)
    groups = any_of or []
    satisfied = 0
    for group in groups:
        options = group if isinstance(group, list) else [group]
        if any((root / opt).exists() for opt in options):
            satisfied += 1
    problems = []
    if missing:
        problems.append("missing: " + ", ".join(missing))
    if empty:
        problems.append("too small: " + ", ".join(empty))
    if groups and satisfied < len(groups):
        problems.append(f"{satisfied}/{len(groups)} alternative file groups satisfied")
    counts = {"files": sum(1 for p in root.rglob("*") if p.is_file()), "present": present}
    if problems:
        return False, "; ".join(problems), counts
    return True, f"{counts['files']} generated file(s), all required entrypoints present", counts


@register_check("media_report")
def _media_report(ctx: VerifyContext, path: str, keys: Any = None, min_chars: int = 0, **_: Any):
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"{path} is not valid JSON: {exc}", str(target)
    if isinstance(keys, str):
        keys = [keys]
    missing = [k for k in (keys or []) if k not in data]
    if missing:
        return False, f"{path} missing observations: {', '.join(missing)}", list(data)[:12]
    transcript = str(data.get("transcript") or "")
    if int(min_chars) and len(transcript) < int(min_chars):
        return False, f"transcript has {len(transcript)} chars, required {min_chars}", len(transcript)
    return True, f"{path} contains {len(data)} observation group(s)", list(data)[:12]


@register_check("markdown_report")
def _markdown_report(ctx: VerifyContext, path: str, min_headings: int = 1, **_: Any):
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    text = target.read_text(encoding="utf-8", errors="replace")
    headings = [line for line in text.splitlines() if line.startswith("#")]
    if len(headings) < int(min_headings):
        return False, f"{path} has {len(headings)} heading(s), required {min_headings}", len(headings)
    return True, f"{path} has {len(headings)} headings and {len(text)} chars", len(headings)


# ----------------------------------------------------------------------
# verifier
# ----------------------------------------------------------------------
class Verifier:
    """Runs checks against an objective and records the evidence."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)

    # -- single check ---------------------------------------------------
    def run_check(self, check: Check, ctx: VerifyContext) -> Evidence:
        fn = _REGISTRY.get(check.kind)
        if fn is None:
            raise CheckError(f"unknown check kind: {check.kind}")
        try:
            ok, detail, observed = fn(ctx, **(check.params or {}))
        except Exception as exc:  # a broken check must never crash the loop
            ok, detail, observed = False, f"check raised {type(exc).__name__}: {exc}", None
        return Evidence(
            check=check.describe(),
            ok=bool(ok),
            detail=str(detail),
            observed=observed,
        )

    # -- requirement ----------------------------------------------------
    def verify_requirement(self, requirement: Requirement, objective: Objective) -> Requirement:
        ctx = VerifyContext(workspace=self.workspace, objective=objective)
        if not requirement.checks:
            requirement.status = Status.DONE if requirement.verified else Status.FAILED
            return requirement
        evidence = [self.run_check(check, ctx) for check in requirement.checks]
        requirement.evidence.extend(evidence)
        requirement.status = Status.DONE if all(e.ok for e in evidence) else Status.FAILED
        return requirement

    def verify_task(self, task: Task, objective: Objective) -> list[Evidence]:
        ctx = VerifyContext(workspace=self.workspace, objective=objective, task=task)
        return [self.run_check(check, ctx) for check in task.checks]

    # -- objective ------------------------------------------------------
    def verify_objective(self, objective: Objective) -> dict:
        results = {}
        for requirement in objective.requirements:
            self.verify_requirement(requirement, objective)
            results[requirement.id] = requirement.verified
        passed = sum(1 for ok in results.values() if ok)
        return {
            "checked": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results,
            "ok": passed == len(results) and bool(results),
        }


@register_check("pptx_notes")
def _pptx_notes(ctx: VerifyContext, path: str, min_with_notes: int = 1, **_: Any):
    target = ctx.resolve(path)
    if not target.exists():
        return False, f"missing file: {path}", str(target)
    try:
        from pptx import Presentation  # type: ignore

        deck = Presentation(str(target))
    except Exception as exc:
        return False, f"cannot open deck {path}: {exc}", str(target)
    with_notes = 0
    for slide in deck.slides:
        try:
            text = slide.notes_slide.notes_text_frame.text.strip()
        except Exception:
            text = ""
        if text:
            with_notes += 1
    ok = with_notes >= int(min_with_notes)
    return ok, f"{with_notes} of {len(deck.slides)} slide(s) have speaker notes (need {min_with_notes})", with_notes
