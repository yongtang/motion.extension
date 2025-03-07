import omni.ext
import omni.kit.app
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

        loop = asyncio.get_event_loop()
        self.server = asyncio.run_coroutine_threadsafe(f(), loop)
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        async def f(task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(f(getattr(self, "server", None)), loop)
        self.server = None
        print("[MotionExtension] Extension shutdown")
