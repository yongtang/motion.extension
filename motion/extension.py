import omni.ext
import omni.kit.app
import asyncio, websockets, toml, os


class MotionExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        server = "ws://localhost:8081"
        try:
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_id = ext_manager.get_extension_id_by_module(__name__)
            ext_path = ext_manager.get_extension_path(ext_id)
            config = os.path.join(ext_path, "config", "config.toml")
            print("[MotionExtension] Extension config: {}".format(config))
            config = toml.load(config)
            print("[MotionExtension] Extension config: {}".format(config))
            server = config.get("server", server) or server
        except Exception as e:
            print("[MotionExtension] Extension config: {}".format(e))
        print("[MotionExtension] Extension server: {}".format(server))

        self.server = server

    def on_startup(self, ext_id):
        async def f():
            while self.running:
                try:
                    async with websockets.connect(self.server) as ws:
                        await ws.send(toml.dumps({"op": "connect", "verbose": True}))
                        await ws.send(
                            toml.dumps(
                                {"op": "sub", "subject": "test.subject", "sid": 1}
                            )
                        )
                        while self.running:
                            try:
                                response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                                print(
                                    "[MotionExtension] Extension server: {}".format(
                                        response
                                    )
                                )
                            except asyncio.TimeoutError:
                                pass
                except Exception as e:
                    print("[MotionExtension] Extension server: {}".format(e))
                    await asyncio.sleep(1)

        self.running = True
        loop = asyncio.get_event_loop()
        self.server_task = loop.create_task(f())
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        async def f():
            if (
                getattr(self, "server_task")
                and self.server_task
                and not self.server_task.done()
            ):
                self.server_task.cancel()
                try:
                    await self.server_task
                except Exception as e:
                    print("[MotionExtension] Extension cancel")

        self.running = False
        loop = asyncio.get_event_loop()
        loop.run_until_complete(f())
        print("[MotionExtension] Extension shutdown")
