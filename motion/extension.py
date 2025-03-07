import omni.ext
import asyncio, uvicorn, multiprocessing
from . import websocket


class MotionExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        def f(data):
            websocket.app.extra["data"] = data
            uvicorn.run(websocket.app, host="0.0.0.0", port=5000, log_level="info")

        self.data = multiprocessing.Value("data", "")
        self.server = multiprocessing.Process(target=f, args=(self.data,), daemon=True)
        self.server.start()
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        if getattr(self, "server") and self.server and self.server.is_alive():
            self.server.terminate()
            self.server.join()
            self.server = None
        print("[MotionExtension] Extension shutdown")
