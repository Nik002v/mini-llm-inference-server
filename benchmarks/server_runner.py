"""Internal local benchmark helper. Stop file requests normal uvicorn shutdown."""

import asyncio
from pathlib import Path
import sys
import uvicorn


async def main():
    server = uvicorn.Server(
        uvicorn.Config(
            "app.main:app",
            host="127.0.0.1",
            port=int(sys.argv[1]),
            log_level="warning",
            timeout_graceful_shutdown=150,
        )
    )
    stopfile = Path(sys.argv[2])

    async def monitor():
        while not stopfile.exists():
            await asyncio.sleep(0.2)
        server.should_exit = True

    watcher = asyncio.create_task(monitor())
    try:
        await server.serve()
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
