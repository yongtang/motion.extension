import omni.ext
import asyncio, uvicorn
from . import websocket


class MotionExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.server = asyncio.get_event_loop().create_task(
            uvicorn.Server(
                uvicorn.Config(
                    websocket.app,
                    host="0.0.0.0",
                    port=5000,
                    log_level="info",
                    loop="asyncio",
                )
            ).serve()
        )
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        async def f(task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.get_event_loop().run_until_complete(f(getattr(self, "server", None)))
        self.server = None
        print("[MotionExtension] Extension shutdown")
