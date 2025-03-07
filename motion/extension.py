import omni.ext
import asyncio, uvicorn
from . import websocket


class MotionExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        async def f():
            try:
                config = uvicorn.Config(
                    websocket.app,
                    host="0.0.0.0",
                    port=5000,
                    log_level="info",
                    loop="asyncio",
                )
                server = uvicorn.Server(config)
                await server.serve()
            except asyncio.CancelledError:
                print("[MotionExtension] Websocket cancel")
                raise

        self.server = asyncio.create_task(f())
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        async def f(task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.ensure_future(f(getattr(self, "server", None)))
        self.server = None
        print("[MotionExtension] Extension shutdown")
