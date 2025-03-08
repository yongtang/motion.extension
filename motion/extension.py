import omni.ext
import omni.kit.app
import asyncio, nats, toml, os


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
        async def message(e):
            print("[MotionExtension] Extension server: {}".format(e))

        async def f():
            while self.running:
                try:
                    nc = await nats.connect("ws://localhost:8081")  # Ensure this is the correct WebSocket URL
                    await nc.subscribe("test.subject", cb=message)
                    await asyncio.sleep(10)
                except Exception as e:
                    print("[MotionExtension] Extension server: {}".format(e))
                    await asyncio.sleep(1)
                finally:
                    await nc.close()


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
