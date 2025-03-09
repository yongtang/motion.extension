import omni.ext
import omni.kit.app
import asyncio, websockets, toml, json, os


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
            try:
                while self.running:
                    try:
                        async with websockets.connect(self.server) as ws:
                            await ws.send("SUB test.subject 1\r\n")
                            while self.running:
                                try:
                                    response = await asyncio.wait_for(
                                        ws.recv(), timeout=1.0
                                    )
                                    print(
                                        "[MotionExtension] Extension server: {}".format(
                                            response
                                        )
                                    )
                                    head, body = response.split(b"\r\n", 1)
                                    if head.startswith(b"MSG "):
                                        assert body.endswith(b"\r\n")
                                        body = body[:-2]

                                        op, sub, sid, count = head.split(b" ", 3)
                                        assert op == b"MSG"
                                        assert sub
                                        assert sid
                                        assert int(count) == len(body)

                                        data = json.loads(body)
                                        print(
                                            "[MotionExtension] Extension server: {}".format(
                                                data
                                            )
                                        )

                                except asyncio.TimeoutError:
                                    pass
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print("[MotionExtension] Extension server: {}".format(e))
                        await asyncio.sleep(1)
            except asyncio.CancelledError:
                print("[MotionExtension] Extension server cancel")
            finally:
                print("[MotionExtension] Extension server exit")

        self.running = True
        loop = asyncio.get_event_loop()
        self.server_task = loop.create_task(f())
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        async def f():
            if getattr(self, "server_task") and self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    print("[MotionExtension] Extension cancel")
                except Exception as e:
                    print("[MotionExtension] Extension exception {}".format(e))

        self.running = False
        loop = asyncio.get_event_loop()
        loop.run_until_complete(f())
        print("[MotionExtension] Extension shutdown")
