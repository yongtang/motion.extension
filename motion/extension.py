import asyncio
import omni.ext
from . import websocket


class MotionExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self.ws_server = websocket.WebSocketServer()

    def on_startup(self, ext_id):
        loop = asyncio.get_event_loop()
        loop.create_task(self.ws_server.start())
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.ws_server.stop())
        print("[MotionExtension] Extension shutdown")
