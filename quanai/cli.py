"""Command line interface.

  quanai run "build a 8-slide deck about solar energy"
  quanai plan "make a mobile app for tracking habits"
  quanai status
  quanai resume obj-1a2b3c4d
  quanai tools
  quanai doctor
  quanai ingest path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .core.agent import Agent
from .core.ledger import Ledger
from .core.memory import Memory
from .core.planner import Planner
from .core.task import Objective
from .llm.router import build_provider, detect_providers
from .tools import build_default_registry, probe_capabilities
from .util import human_bytes, now_iso


def _config(args) -> Config:
    overrides = {
        "workspace": getattr(args, "workspace", None),
        "home": getattr(args, "home", None),
        "max_iterations": getattr(args, "max_iterations", None),
        "provider": getattr(args, "provider", None),
        "image_backend": getattr(args, "image_backend", None),
    }
    config = Config.from_env(**{k: v for k, v in overrides.items() if v is not None})
    return config.ensure_dirs()


def cmd_run(args) -> int:
    config = _config(args)
    agent = Agent(config)
    report = agent.run(
        args.request,
        objective_id=args.id,
        max_iterations=args.max_iterations,
        attach=args.input or None,
    )
    print(report.summary())
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0 if report.verified else 2


def cmd_plan(args) -> int:
    config = _config(args)
    provider = build_provider(config)
    registry = build_default_registry()
    objective = Objective.new(args.request, workspace=str(config.workspace))
    plan = Planner(config, registry, provider, Memory(config.home / "memory")).plan(objective)

    print(f"Request: {args.request}")
    print(f"Plan source: {plan.source}")
    print(f"Intents: {', '.join(plan.thought.get('intents', []))}")
    print(f"Chosen: {'; '.join(plan.thought.get('chosen', [])) or 'n/a'}")
    print("\nRequirements:")
    for req in plan.requirements:
        flag = "required" if req.required else "optional"
        print(f"  {req.id} ({flag}): {req.text}")
        for check in req.checks:
            print(f"      checks: {check.describe()}")
    print("\nTasks:")
    for task in plan.tasks:
        deps = f"  after {', '.join(task.depends_on)}" if task.depends_on else ""
        print(f"  {task.id}: {task.tool} - {task.title}{deps}")
        print(f"      args: {json.dumps(task.args)[:220]}")
    if plan.notes:
        print("\nNotes:")
        for note in plan.notes:
            print(f"  - {note}")
    return 0


def cmd_status(args) -> int:
    config = _config(args)
    ledger = Ledger(config.home)
    rows = ledger.summary()
    if not rows:
        print("No objectives yet. Try: quanai run \"build a deck about ...\"")
        return 0
    print(f"{'id':22} {'status':9} {'done%':>6}  request")
    for row in rows[: args.limit]:
        print(f"{row['id']:22} {row['status']:9} {row['completion']:>6}  {(row['request'] or '')[:70]}")
    memory = Memory(config.home / "memory")
    stats = memory.stats()
    print(
        f"\nstate: {config.home}  |  memory items: {stats['items']}  "
        f"lessons: {stats['lessons']}  plan templates: {stats['plans']}"
    )
    return 0


def cmd_show(args) -> int:
    config = _config(args)
    ledger = Ledger(config.home)
    objective = ledger.load(args.id)
    if objective is None:
        print(f"No objective {args.id}", file=sys.stderr)
        return 1
    print(json.dumps(objective.to_dict(), indent=2, default=str))
    return 0


def cmd_resume(args) -> int:
    config = _config(args)
    agent = Agent(config)
    report = agent.resume(args.id, max_iterations=args.max_iterations)
    print(report.summary())
    return 0 if report.verified else 2


def cmd_tools(args) -> int:
    registry = build_default_registry()
    print(f"{len(registry.names())} tools registered\n")
    for spec in registry.describe():
        required = ", ".join(spec["required"]) or "-"
        print(f"{spec['name']:22} [{spec['category']}] {spec['summary']}")
        print(f"{'':22} required: {required}  needs: {', '.join(spec['requires']) or '-'}")
    return 0


def cmd_doctor(args) -> int:
    config = _config(args)
    caps = probe_capabilities()
    print("Capabilities")
    for key, value in caps.items():
        mark = "ok" if value else "--"
        print(f"  [{mark}] {key:16} {value or 'not available'}")

    print("\nLanguage model")
    provider = build_provider(config)
    if provider is None:
        print("  offline deterministic engine (no model configured)")
        print("  to use a local model:  ollama serve  &&  QUANAI_LLM_PROVIDER=ollama quanai run ...")
        print("  to use a hosted model: export QUANAI_API_KEY=... QUANAI_MODEL=gpt-4o-mini")
    else:
        print(f"  {provider.name}:{provider.model} @ {getattr(provider, 'base_url', '-')}")
        try:
            from .llm.base import Message

            completion = provider.complete([Message.user("Reply with the single word: ready")], max_tokens=10)
            print(f"  live check: {completion.text[:60]!r} ({completion.latency}s)")
        except Exception as exc:
            print(f"  live check failed: {exc}")

    if args.all:
        print("\nBackends probed")
        for row in detect_providers(config):
            state = "reachable" if row["reachable"] else "not found"
            print(f"  {row['provider']:10} {row['base_url']:38} {row['model']:20} {state} ({row['cost']})")

    print(f"\nWorkspace: {config.workspace}")
    print(f"State:     {config.home}")
    print(f"Image backend: {config.image_backend}   network: {config.allow_network}   shell: {config.allow_shell}")
    return 0


def cmd_ingest(args) -> int:
    config = _config(args)
    from .tools import ToolContext
    from .tools.media import register as register_media
    from .tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_media(registry)
    ctx = ToolContext(workspace=config.workspace, config=config, provider=build_provider(config))
    result = registry.execute(
        "media.ingest",
        {
            "path": args.path,
            "out_dir": args.out_dir or f"media/{Path(args.path).stem}",
            "frames": args.frames,
            "transcribe": not args.no_transcribe,
            "describe": not args.no_describe,
        },
        ctx,
    )
    if not result.ok:
        print(f"ingest failed: {result.error}", file=sys.stderr)
        return 1
    print(json.dumps(result.data, indent=2, default=str))
    for art in result.artifacts:
        print(f"  {art.path}  ({human_bytes(art.bytes)})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quanai",
        description="Self-checking multimodal agent: decks, images, video, apps and media analysis.",
    )
    parser.add_argument("--workspace", help="where artifacts are written")
    parser.add_argument("--home", help="where agent state is stored")
    parser.add_argument("--provider", help="offline | auto | ollama | openai | lmstudio | vllm | <custom>")
    parser.add_argument("--image-backend", help="auto | procedural | sd | pollinations")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan, execute and verify a request")
    run.add_argument("request")
    run.add_argument("--id", help="objective id (default: generated)")
    run.add_argument("--input", action="append", help="attach an input file (image/audio/video)")
    run.add_argument("--max-iterations", type=int, default=None)
    run.add_argument("--json", action="store_true", help="also print the full report as JSON")
    run.set_defaults(func=cmd_run)

    plan = sub.add_parser("plan", help="show the requirements and task graph without running it")
    plan.add_argument("request")
    plan.set_defaults(func=cmd_plan)

    status = sub.add_parser("status", help="list objectives")
    status.add_argument("--limit", type=int, default=20)
    status.set_defaults(func=cmd_status)

    show = sub.add_parser("show", help="dump one objective as JSON")
    show.add_argument("id")
    show.set_defaults(func=cmd_show)

    resume = sub.add_parser("resume", help="continue an interrupted objective")
    resume.add_argument("id")
    resume.add_argument("--max-iterations", type=int, default=None)
    resume.set_defaults(func=cmd_resume)

    tools = sub.add_parser("tools", help="list the tool surface")
    tools.set_defaults(func=cmd_tools)

    doctor = sub.add_parser("doctor", help="check this machine and the model setup")
    doctor.add_argument("--all", action="store_true", help="probe every known backend")
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest", help="analyse an image, audio or video file")
    ingest.add_argument("path")
    ingest.add_argument("--out-dir")
    ingest.add_argument("--frames", type=int, default=6)
    ingest.add_argument("--no-transcribe", action="store_true")
    ingest.add_argument("--no-describe", action="store_true")
    ingest.set_defaults(func=cmd_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted - progress is saved, resume with: quanai resume <id>", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
