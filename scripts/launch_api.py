#!/usr/bin/env python3
"""
PHILOS — Launch API Server.

Starts the FastAPI server for inter-module communication.

Usage:
    python scripts/launch_api.py
    python scripts/launch_api.py --port 8000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHILOS API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run(
        "philos.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
