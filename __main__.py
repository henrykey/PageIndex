#!/usr/bin/env python3
"""Entry point for PageIndex MCP service."""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PageIndex MCP service")
    parser.add_argument("--http", action="store_true", help="Run with streamable HTTP transport")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9011)
    return parser.parse_args()


async def async_main() -> None:
    args = _parse_args()
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    from .server import mcp

    if args.http:
        await mcp.run_async(transport="streamable-http", host=args.host, port=args.port)
    else:
        await mcp.run_async()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
