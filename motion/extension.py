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
        if getattr(self, "server", None) and not self.server.done():
            self.server.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self.server)
            except asyncio.CancelledError:
                print("[MotionExtension] Server shutdown")
            except RuntimeError as e:
                print("[MotionExtension] Server exception: {}".format(e))
            self.server = None
        print("[MotionExtension] Extension shutdown")
