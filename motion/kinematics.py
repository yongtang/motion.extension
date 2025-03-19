import omni.ext
import omni.usd
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.universal_robots import UR10

# from omni.isaac.core.articulations import Articulation
from omni.isaac.universal_robots.kinematics_solver import KinematicsSolver
from scipy.spatial.transform import Rotation as R
import asyncio, websockets, toml, json, os
import numpy as np


class MotionKinematicsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self.config = {
            "subject": "subject.pose",
            "effector": None,
            "articulation": None,
            "server": "ws://localhost:8081",
        }

        try:
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_id = ext_manager.get_extension_id_by_module(__name__)
            ext_path = ext_manager.get_extension_path(ext_id)
            config = os.path.join(ext_path, "config", "config.toml")
            print("[MotionKinematicsExtension] Extension config: {}".format(config))
            config = toml.load(config)
            print("[MotionKinematicsExtension] Extension config: {}".format(config))
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
            print("[MotionKinematicsExtension] Extension config: {}".format(e))
        print("[MotionKinematicsExtension] Extension config: {}".format(self.config))

        # step => pose - pose(last) + link, delta = link - pose(last)
        self.kinematics_pose = None
        self.kinematics_delta = None

    def on_startup(self, ext_id):
        async def f(self):
            context = omni.usd.get_context()
            while context.get_stage() is None:
                print("[MotionKinematicsExtension] Extension world wait")
                await asyncio.sleep(0.5)
            print("[MotionKinematicsExtension] Extension world ready")

            stage = context.get_stage()
            print("[MotionKinematicsExtension] Extension stage {}".format(stage))

            if self.config["articulation"]:
                self.articulation = Articulation(self.config["articulation"])
                while not self.articulation.is_initialized():
                    print("[MotionKinematicsExtension] Extension articulation wait")
                    await asyncio.sleep(0.5)

                self.controller = self.articulation.get_articulation_controller()
                self.solver = KinematicsSolver(
                    self.articulation, self.config["effector"]
                )
                print(
                    "[MotionKinematicsExtension] Extension articulation {} ({}) {} {}".format(
                        self.articulation,
                        self.articulation.dof_names,
                        self.controller,
                        self.solver,
                    )
                )

        async def g(self):
            try:
                while self.running:
                    try:
                        async with websockets.connect(self.config["server"]) as ws:
                            await ws.send("SUB {} 1\r\n".format(self.config["subject"]))
                            while self.running:
                                try:
                                    response = await asyncio.wait_for(
                                        ws.recv(), timeout=1.0
                                    )
                                    print(
                                        "[MotionKinematicsExtension] Extension server: {}".format(
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

                                        self.kinematics_pose = json.loads(body)
                                        print(
                                            "[MotionKinematicsExtension] Extension server pose: {}".format(
                                                self.kinematics_pose
                                            )
                                        )

                                except asyncio.TimeoutError:
                                    pass
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print(
                            "[MotionKinematicsExtension] Extension server: {}".format(e)
                        )
                        await asyncio.sleep(1)
            except asyncio.CancelledError:
                print("[MotionKinematicsExtension] Extension server cancel")
            finally:
                print("[MotionKinematicsExtension] Extension server exit")

        self.kinematics_pose = None
        self.kinematics_delta = None

        self.running = True
        loop = asyncio.get_event_loop()
        loop.run_until_complete(f(self))
        self.server_task = loop.create_task(g(self))
        print("[MotionKinematicsExtension] Extension startup")

        self.subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self.world_callback, name="world_callback")
        )

    def world_callback(self, e):
        try:
            world = World.instance()
            if world and world.stage:
                print(
                    "[MotionKinematicsExtension] Extension world: {} {}".format(
                        world, world.stage
                    )
                )
                if world.physics_callback_exists("on_physics_step"):
                    world.remove_physics_callback("on_physics_step")
                world.add_physics_callback("on_physics_step", self.on_physics_step)
                self.subscription = None
                # from .simple_stack import SimpleStack

                # self.stack = SimpleStack(world=world)
                # self.stack.setup_post_load()
                return
        except Exception as e:
            print("[MotionKinematicsExtension] Extension world: {}".format(e))

    def on_physics_step(self, step_size):
        pose = self.kinematics_pose
        if pose is None:
            return
        if (
            self.kinematics_delta is None
            or self.kinematics_delta["channel"] != pose["channel"]
        ):
            # step => pose - pose(last) + link, delta = link - pose(last)
            position, orientation = XFormPrim(
                "{}/{}".format(self.config["articulation"], self.config["effector"])
            ).get_world_pose()

            print(
                "[MotionKinematicsExtension] Extension reference pose: {} {}".format(
                    position, orientation
                )
            )

            delta_p = position - np.array(
                (
                    pose["position"]["x"],
                    pose["position"]["y"],
                    pose["position"]["z"],
                )
            )
            delta_o = (
                R.from_quat(
                    np.array(
                        (
                            orientation[1],
                            orientation[2],
                            orientation[3],
                            orientation[0],
                        )
                    )
                )
                * R.from_quat(
                    np.array(
                        (
                            pose["orientation"]["x"],
                            pose["orientation"]["y"],
                            pose["orientation"]["z"],
                            pose["orientation"]["w"],
                        )
                    )
                ).inv()
            ).as_quat()

            self.kinematics_delta = {
                "position": {
                    "x": delta_p[0],
                    "y": delta_p[1],
                    "z": delta_p[2],
                },
                "orientation": {
                    "x": delta_o[0],
                    "y": delta_o[1],
                    "z": delta_o[2],
                    "w": delta_o[3],
                },
                "channel": pose["channel"],
            }
            print(
                "[MotionKinematicsExtension] Extension reference delta: {} {}".format(
                    self.kinematics_delta
                )
            )
            return

        # step => pose - pose(last) + link, delta = link - pose(last), step => pose + delta
        delta = self.kinematics_delta
        position = np.array(
            (
                pose["position"]["x"],
                pose["position"]["y"],
                pose["position"]["z"],
            )
        ) + np.array(
            (
                delta["position"]["x"],
                delta["position"]["y"],
                delta["position"]["z"],
            )
        )
        orientation = (
            R.from_quat(
                np.array(
                    (
                        delta["orientation"]["x"],
                        delta["orientation"]["y"],
                        delta["orientation"]["z"],
                        delta["orientation"]["w"],
                    )
                )
            )
            * R.from_quat(
                np.array(
                    (
                        pose["orientation"]["x"],
                        pose["orientation"]["y"],
                        pose["orientation"]["z"],
                        pose["orientation"]["w"],
                    )
                )
            )
        ).as_quat()

        target_position = position
        target_orientation = np.array(
            (orientation[3], orientation[0], orientation[1], orientation[2])
        )

        kinematics, success = self.solver.compute_inverse_kinematics(
            target_position=target_position, target_orientation=target_orientation
        )
        print(
            "[MotionKinematicsExtension] Extension kinematics: {} {}".format(
                kinematics, success
            )
        )

        if success:
            self.controller.apply_action(kinematics)

        # self.stack.call_physics_step(position, orientation, step_size)

        return

    def on_shutdown(self):

        self.subscription = None

        self.kinematics_pose = None
        self.kinematics_delta = None

        async def g(self):
            if getattr(self, "server_task") and self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    print("[MotionKinematicsExtension] Extension server cancel")
                except Exception as e:
                    print(
                        "[MotionKinematicsExtension] Extension server exception {}".format(
                            e
                        )
                    )

        self.running = False
        loop = asyncio.get_event_loop()
        loop.run_until_complete(g(self))
        print("[MotionKinematicsExtension] Extension shutdown")
