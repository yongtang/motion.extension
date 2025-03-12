import omni.ext
import omni.usd
import omni.kit.app
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.dynamic_control import _dynamic_control
from scipy.spatial.transform import Rotation as R
import asyncio, websockets, toml, json, os, socket
import numpy as np


class MotionExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        camera = None
        articulation = None
        server = "ws://localhost:8081"
        effector = None
        try:
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_id = ext_manager.get_extension_id_by_module(__name__)
            ext_path = ext_manager.get_extension_path(ext_id)
            config = os.path.join(ext_path, "config", "config.toml")
            print("[MotionExtension] Extension config: {}".format(config))
            config = toml.load(config)
            print("[MotionExtension] Extension config: {}".format(config))
            camera = config.get("camera", camera) or camera
            articulation = config.get("articulation", articulation) or articulation
            server = config.get("server", server) or server
            effector = config.get("effector", effector) or effector
        except Exception as e:
            print("[MotionExtension] Extension config: {}".format(e))
        print("[MotionExtension] Extension camera: {}".format(camera))
        print("[MotionExtension] Extension articulation: {}".format(articulation))
        print("[MotionExtension] Extension server: {}".format(server))
        print("[MotionExtension] Extension effector: {}".format(effector))

        self.camera = camera
        self.articulation_path = articulation_path
        self.server = server
        self.effector = effector
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def on_startup(self, ext_id):
        async def f(self):
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

        async def g(self):
            context = omni.usd.get_context()
            while context.get_stage() is None:
                print("[MotionExtension] Extension world wait")
                await asyncio.sleep(0.5)
            print("[MotionExtension] Extension world ready")

            stage = context.get_stage()
            print("[MotionExtension] Extension stage {}".format(stage))

            if self.camera and stage.GetPrimAtPath(self.camera).IsValid():
                self.camera = Camera(prim_path=self.camera)
                self.camera.initialize()
                print("[MotionExtension] Extension camera {}".format(self.camera))
            else:
                self.camera = None

            if self.articulation_path:
                self.articulation = Articulation(self.articulation_path)
                print(
                    "[MotionExtension] Extension articulation {} ({})".format(
                        self.articulation, self.articulation.dof_names
                    )
                )
                self.dynamic_control = (
                    _dynamic_control.acquire_dynamic_control_interface()
                )
                print(
                    "[MotionExtension] Extension dynamic_control {}".format(
                        self.dynamic_control
                    )
                )
                self.articulation_prim = self.dynamic_control.get_articulation(
                    self.articulation_path
                )

        async def v(self):
            try:
                while self.running:
                    try:
                        image = self.camera.get_rgba()
                        if len(image):
                            image = np.array(image, dtype=np.uint8)  # RGBA
                            assert image.shape[-1] == 4, "camera {}".format(image.shape)
                            image = image[:, :, :3][:, :, ::-1]  # BGR
                            assert image.shape[-1] == 3, "camera {}".format(image.shape)
                            self.socket.sendto(image.tobytes(), ("127.0.0.1", 6000))
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
        loop.run_until_complete(g(self))
        self.server_task = loop.create_task(f(self))
        self.camera_task = loop.create_task(v(self)) if self.camera else None
        print("[MotionExtension] Extension startup")

        self.subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self.step, name="StepFunction")
        )

    def step(self, delta: float):
        print("[MotionExtension] Extension step {}".format(delta))

        def apply_differential_ik(delta_p, delta_r, delta_t):
            """
            Apply Differential IK (Jacobian-based) to move the end-effector smoothly over time.

            Arguments:
            - delta_p: [dx, dy, dz] position displacement
            - delta_r: [dqx, dqy, dqz, dqw] quaternion rotation change
            - delta_t: time step (seconds)
            """
            # Get current end effector pose
            # Get the articulation handle
            articulation = self.dynamic_control.get_articulation(self.articulation_path)

            link_name = self.effector
            # Iterate through all links to find the desired one
            for i in range(
                self.dynamic_control.get_articulation_link_count(articulation)
            ):
                link = self.dynamic_control.get_articulation_link(articulation, i)
                if (
                    self.dynamic_control.get_articulation_link_name(articulation, link)
                    == link_name
                ):
                    # Get the pose of the link in the world frame
                    pose = self.dynamic_control.get_rigid_body_pose(link)
                    current_ee_pos = pose.p
                    current_ee_rot = pose.r
                    print(f"Position: {position}, Orientation: {orientation}")
                    break

            # Compute linear velocity
            linear_velocity = delta_p / delta_t  # [vx, vy, vz]

            # Compute current and desired rotations
            current_rot = R.from_quat(current_ee_rot)  # Current rotation
            desired_rot = R.from_quat(delta_r)  # Desired rotation

            # Compute relative rotation (delta rotation)
            delta_rot = desired_rot * current_rot.inv()

            # Convert delta rotation to angular velocity
            angular_velocity = delta_rot.as_rotvec() / delta_t  # [wx, wy, wz]

            # Combine linear and angular velocities
            cartesian_velocity = np.hstack(
                (linear_velocity, angular_velocity)
            )  # Shape: (6,)

            # Compute Jacobian matrix
            jacobian_matrix = self.articulation.compute_jacobian(
                effector
            )  # Shape: (6, num_joints)

            # Compute pseudoinverse of the Jacobian
            jacobian_pinv = np.linalg.pinv(jacobian_matrix)  # Shape: (num_joints, 6)

            # Compute joint velocities
            joint_velocities = (
                jacobian_pinv @ cartesian_velocity
            )  # Shape: (num_joints,)

            # Apply joint velocities to the robot
            # articulation_prim = dc.get_articulation(robot_prim_path)
            # dc.set_articulation_dof_velocity_targets(articulation_prim, joint_velocities)

            # Do NOT call `dc.step(delta_t)` in an extension! The simulation loop handles stepping.

            return joint_velocities

        if self.position is not None:
            if self.step_position is not None:
                delta_p = self.position[:3] - self.step_position[:3]
                delta_r = (
                    R.from_quat(self.position[3:])
                    * R.from_quat(self.step_position[3:]).inv()
                ).as_quat()

                joint_velocities = apply_differential_ik(delta_p, delta_r, delta)

                articulation_prim = self.dynamic_control.get_articulation(
                    robot_prim_path
                )
                self.dynamic_control.set_articulation_dof_velocity_targets(
                    articulation_prim, joint_velocities
                )

                print("[MotionExtension] Extension {}".format(joint_velocities))
            self.step_position = self.position

    def on_shutdown(self):
        async def f(self):
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
        loop.run_until_complete(f(self))
        loop.run_until_complete(v(self))
        print("[MotionExtension] Extension shutdown")
