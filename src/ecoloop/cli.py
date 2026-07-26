"""Eco-Loop CLI entry point."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="ecoloop", description="Eco-Loop Building Agents")
    sub = parser.add_subparsers(dest="command", required=True)

    # spike - runtime actuation proof
    spike = sub.add_parser("spike", help="Run the runtime callback spike")
    spike.add_argument("--idf", default=r"D:\EnergyPlus\ExampleFiles\5ZoneAirCooled.idf")
    spike.add_argument("--epw", default=r"D:\EnergyPlus\WeatherData\USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw")
    spike.add_argument("--output", default="data/output/runtime-spike")

    # demo - offline deterministic control loop
    demo = sub.add_parser("demo", help="Run offline deterministic control loop demo")
    demo.add_argument("--mode", choices=["ai", "baseline", "compare"], default="compare")
    demo.add_argument("--steps", type=int, default=96, help="Number of 15-minute steps to simulate (96 = 24h)")

    # run - full simulation with controller
    run = sub.add_parser("run", help="Run full simulation with controller")
    run.add_argument("--mode", choices=["ai", "baseline", "compare"], default="compare")
    run.add_argument("--steps", type=int, default=96, help="Number of 15-minute steps to simulate (96 = 24h)")
    run.add_argument("--idf", default=r"D:\EnergyPlus\ExampleFiles\5ZoneAirCooled.idf")
    run.add_argument("--epw", default=r"D:\EnergyPlus\WeatherData\USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    # mcp - Model Context Protocol server
    mcp_parser = sub.add_parser("mcp", help="Start MCP server (stdio or HTTP)")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
    )
    mcp_parser.add_argument("--host", default="127.0.0.1")
    mcp_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "spike":
        from .runtime import run_schedule_actuation_spike
        result = run_schedule_actuation_spike(args.idf, args.epw, args.output)
        print(json.dumps(result.__dict__, indent=2))
        sys.exit(result.exit_code)

    elif args.command == "demo":
        from .controller import ClosedLoopController
        from .config import RunConfig
        controller = ClosedLoopController(RunConfig())
        if args.mode == "compare":
            metrics = controller.compare(steps=args.steps)
            print("\n=== Comparison Results ===")
            print(json.dumps(metrics, indent=2))
        else:
            events = controller.run(mode=args.mode, steps=args.steps)
            print(f"\n[OK] {args.mode} run completed: {len(events)} events")

    elif args.command == "run":
        from .controller import ClosedLoopController
        from .config import RunConfig
        config = RunConfig()
        controller = ClosedLoopController(
            config,
            idf_path=Path(args.idf),
            epw_path=Path(args.epw),
        )
        if args.mode == "compare":
            metrics = controller.compare()
            print("\n=== Comparison Results ===")
            print(json.dumps(metrics, indent=2))
        else:
            events = controller.run(mode=args.mode)
            print(f"\n[OK] {args.mode} run completed: {len(events)} events")

    elif args.command == "dashboard":
        # Launch Streamlit as a subprocess so it gets the proper ScriptRunContext
        script_path = Path(__file__).with_name("dashboard_app.py")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(script_path)],
                check=False,
            )
        except FileNotFoundError:
            print("Error: streamlit not found. Install with: uv add streamlit")
            sys.exit(1)

    elif args.command == "mcp":
        from .mcp_server import run_server
        run_server(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
