import omni.ext
import omni.usd
import omni.kit.app
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.dynamic_control import _dynamic_control
from scipy.spatial.transform import Rotation as R
import asyncio, websockets, toml, json, os, socket, io
import numpy as np
import PIL.Image


class MotionExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self.config = {
            "camera": None,
            "effector": None,
            "articulation": None,
            "server": "ws://localhost:8081",
        }

        try:
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_id = ext_manager.get_extension_id_by_module(__name__)
            ext_path = ext_manager.get_extension_path(ext_id)
            config = os.path.join(ext_path, "config", "config.toml")
            print("[MotionExtension] Extension config: {}".format(config))
            config = toml.load(config)
            print("[MotionExtension] Extension config: {}".format(config))
            self.config["camera"] = (
                config.get("camera", self.config["camera"]) or self.config["camera"]
            )
            self.config["effector"] = (
                config.get("effector", self.config["effector"])
                or self.config["effector"]
            )
            self.config["articulation"] = (
                config.get("articulation", self.config["articulation"])
                or self.config["articulation"]
            )
            self.config["server"] = (
                config.get("server", self.config["server"]) or self.config["server"]
            )
        except Exception as e:
            print("[MotionExtension] Extension config: {}".format(e))
        print("[MotionExtension] Extension config: {}".format(self.config))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def on_startup(self, ext_id):
        async def f(self):
            context = omni.usd.get_context()
            while context.get_stage() is None:
                print("[MotionExtension] Extension world wait")
                await asyncio.sleep(0.5)
            print("[MotionExtension] Extension world ready")

            stage = context.get_stage()
            print("[MotionExtension] Extension stage {}".format(stage))

            if (
                self.config["camera"]
                and stage.GetPrimAtPath(self.config["camera"]).IsValid()
            ):
                self.camera = Camera(prim_path=self.config["camera"])
                self.camera.initialize()
                print("[MotionExtension] Extension camera {}".format(self.camera))
            else:
                self.camera = None

            if self.config["articulation"]:
                self.articulation = Articulation(self.config["articulation"])
                print(
                    "[MotionExtension] Extension articulation {} ({})".format(
                        self.articulation, self.articulation.dof_names
                    )
                )

        async def g(self):
            try:
                while self.running:
                    try:
                        async with websockets.connect(self.config["server"]) as ws:
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
                                        self.position = np.array(
                                            (
                                                data["position"]["x"],
                                                data["position"]["y"],
                                                data["position"]["z"],
                                                data["orientation"]["x"],
                                                data["orientation"]["y"],
                                                data["orientation"]["z"],
                                                data["orientation"]["w"],
                                            )
                                        )
                                        print(
                                            "[MotionExtension] Extension position: {}".format(
                                                self.position
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

        async def v(self):
            try:
                while self.running:
                    try:
                        image = self.camera.get_rgba()
                        if len(image):
                            image = np.array(image, dtype=np.uint8)  # RGBA
                            assert image.shape[-1] == 4, "camera {}".format(image.shape)
                            image = image[:, :, :3]  # Remove alpha channel, keep RGB
                            assert image.shape[-1] == 3, "camera {}".format(image.shape)
                            buffer = io.BytesIO()
                            PIL.Image.fromarray(image, mode="RGB").save(
                                buffer, format="JPEG2000", quality_mode="lossless"
                            )
                            self.socket.sendto(buffer.getvalue(), ("127.0.0.1", 6000))
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print("[MotionExtension] Extension camera: {}".format(e))
                    await asyncio.sleep(1 / 30)  # Maintain ~30 FPS
            except asyncio.CancelledError:
                print("[MotionExtension] Extension camera cancel")
            finally:
                print("[MotionExtension] Extension camera exit")

        self.position = None
        self.step_position = None

        self.running = True
        loop = asyncio.get_event_loop()
        loop.run_until_complete(f(self))
        self.server_task = loop.create_task(g(self))
        self.camera_task = loop.create_task(v(self)) if self.camera else None
        print("[MotionExtension] Extension startup")

        # self.subscription = (
        #    omni.kit.app.get_app()
        #    .get_update_event_stream()
        #    .create_subscription_to_pop(self.step, name="StepFunction")
        # )

    def step(self, delta: float):
        print("[MotionExtension] Extension step {}".format(delta))

        if self.position is not None:
            value = self.position
            if self.step_position is not None:
                delta_p = value[:3] - value[:3]
                delta_r = (
                    R.from_quat(value[3:]) * R.from_quat(value[3:]).inv()
                ).as_quat()
                print(
                    "[MotionExtension] Extension step {} {} {}".format(
                        delta_p, delta_r, delta
                    )
                )
            self.step_position = value

    def on_shutdown(self):
        async def g(self):
            if getattr(self, "server_task") and self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    print("[MotionExtension] Extension server cancel")
                except Exception as e:
                    print("[MotionExtension] Extension server exception {}".format(e))

        async def v(self):
            if getattr(self, "camera_task") and self.camera_task:
                self.camera_task.cancel()
                try:
                    await self.camera_task
                except asyncio.CancelledError:
                    print("[MotionExtension] Extension camera cancel")
                except Exception as e:
                    print("[MotionExtension] Extension camera exception {}".format(e))

        self.running = False
        loop = asyncio.get_event_loop()
        loop.run_until_complete(g(self))
        loop.run_until_complete(v(self))
        print("[MotionExtension] Extension shutdown")
