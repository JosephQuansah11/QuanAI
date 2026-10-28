"""Use QuanAI as a library: run, inspect, resume, and add a custom tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from quanai import Agent, Config  # noqa: E402
from quanai.tools import build_default_registry  # noqa: E402
from quanai.tools.registry import ToolResult  # noqa: E402


def main() -> None:
    config = Config.from_env()
    agent = Agent(config)

    report = agent.run("create a 6 slide deck about urban cycling safety")
    print(report.summary())

    print("\nVerified:", report.verified)
    for requirement in report.requirements:
        state = "PASS" if requirement["verified"] else "FAIL"
        print(f"  [{state}] {requirement['id']}: {requirement['text']}")
        for evidence in requirement["evidence"]:
            print(f"         {evidence['detail']}")

    print("\nArtifacts:")
    for artifact in report.artifacts:
        print(f"  {artifact['path']} ({artifact['bytes']} bytes)")

    print(f"\nFull report: {report.report_path}")

    # Resume is safe even after success: it re-verifies and returns the same state.
    again = Agent(Config.from_env()).resume(report.objective_id)
    assert again.verified == report.verified

    # A custom tool joins the same registry and can be planned like any other.
    registry = build_default_registry()

    @registry.tool(
        "demo.hello",
        "write a greeting file into the workspace",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "name": {"type": "string"}},
            "required": ["path", "name"],
        },
        category="demo",
    )
    def hello(ctx, path: str, name: str) -> ToolResult:
        target = ctx.path(path)
        target.write_text(f"hello {name}\n", encoding="utf-8")
        return ToolResult.success({"path": ctx.rel(target)})

    print("\nCustom tool registered:", "demo.hello" in registry.names())


if __name__ == "__main__":
    main()
