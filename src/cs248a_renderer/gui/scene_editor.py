from typing import Unpack
from imgui_bundle import ImVec4, imgui, imgui_ctx
from pyglm import glm
from reactivex.subject import BehaviorSubject
from slangpy_imgui_bundle.render_targets.window import Window, WindowArgs

from cs248a_renderer.model.transforms import Transform3D
from cs248a_renderer.model.scene import Scene
from cs248a_renderer.model.scene_object import SceneObject
from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.view_model.scene_manager import SceneManager


class SceneEditorArgs(WindowArgs):
    scene_manager: SceneManager
    editing_object: BehaviorSubject[SceneObject | None]
    mesh_outdated: BehaviorSubject[bool]


class SceneEditorWindow(Window):
    _scene_manager: SceneManager
    _editing_object: BehaviorSubject[SceneObject | None]
    _dnd_store: dict[int, str]
    _mesh_outdated: BehaviorSubject[bool]

    def __init__(self, **kwargs: Unpack[SceneEditorArgs]) -> None:
        super().__init__(**kwargs)
        self._scene_manager = kwargs["scene_manager"]
        self._editing_object = kwargs["editing_object"]
        self._mesh_outdated = kwargs["mesh_outdated"]
        self._dnd_store = {}

    def render_window(self, time: float, delta_time: float, open: bool | None) -> bool:
        with imgui_ctx.begin("Scene Editor", p_open=open) as window:
            with imgui_ctx.push_item_width(-150):
                # has_volume_scene = self._scene_manager.volume_scene is not None
                # has_nerf_scene = self._scene_manager.nerf_scene is not None

                self._render_scene_camera(self._scene_manager.scene)
                self._render_scene_graph(self._scene_manager.scene.root)

                # Add new empty SceneObject
                if imgui.button("Add Empty SceneObject"):
                    new_object = SceneObject()
                    self._scene_manager.scene.add_object(new_object)
                    self._mesh_outdated.on_next(True)
                # if not has_volume_scene and not has_nerf_scene:
                #     imgui.text_colored(ImVec4(1.0, 0.0, 0.0, 1.0), "NO SCENE LOADED")
                # else:
                #     if has_volume_scene:
                #         self._render_volume_scene(self._scene_manager.volume_scene)
                #     elif has_nerf_scene:
                #         self._render_nerf_scene(self._scene_manager.nerf_scene)
            return window.opened

    def _render_camera_section(self, camera: PerspectiveCamera, suffix: str) -> None:
        imgui.separator_text(f"{suffix} Camera")
        self._render_transform(camera, f"{suffix} Camera")
        changed, fov = imgui.drag_float(
            f"FOV##{suffix}",
            camera.fov,
            v_speed=0.1,
            v_min=1.0,
            v_max=179.0,
        )
        if changed:
            camera.fov = fov

    def _render_scene_camera(self, scene: Scene) -> None:
        self._render_camera_section(scene.camera, "Scene")

    def _render_scene_graph(self, root: SceneObject) -> None:
        imgui.separator_text("Scene Graph")
        self._render_scene_graph_node(root, is_root=True)

    def _render_scene_graph_node(
        self, node: SceneObject, is_root: bool = False
    ) -> None:
        label = f"{node.name}##SceneGraphNode{node.name}"
        imgui.push_id(node.name)
        if imgui.tree_node(label):
            # Handle drag and drop for reparenting
            if imgui.begin_drag_drop_source():
                payload = node.name
                payload_id = id(payload)
                self._dnd_store[payload_id] = payload
                imgui.set_drag_drop_payload_py_id(
                    "SCENE_OBJECT",
                    payload_id,
                )
                imgui.text(f"Reparent {node.name}")
                imgui.end_drag_drop_source()

            if imgui.begin_drag_drop_target():
                payload_id = imgui.accept_drag_drop_payload_py_id("SCENE_OBJECT")
                if payload_id is not None:
                    print(f"Dropping payload id: {payload_id.data_id}")
                    payload = self._dnd_store.get(payload_id.data_id, None)
                    print(f"Dropping on {node.name}")
                    if payload != node.name:
                        self._scene_manager.scene.reparent(payload, node.name)
                        self._mesh_outdated.on_next(True)
                imgui.end_drag_drop_target()

            if not is_root:
                self._render_transform(node, node.name)
            for child in node.children:
                self._render_scene_graph_node(child)

            if not is_root:
                if imgui.button(f"Delete {node.name}"):
                    self._scene_manager.scene.remove_object(node.name)
                    self._mesh_outdated.on_next(True)
                    imgui.tree_pop()
                    imgui.pop_id()
                    return
            imgui.tree_pop()
        imgui.pop_id()

    def _render_transform(self, node: SceneObject, name: str = "") -> None:
        transform = node.transform
        # Position
        changed, position = imgui.drag_float3(
            f"Position##{name}", transform.position.to_list(), v_speed=0.01
        )
        if changed:
            transform.position = glm.vec3(*position)
            self._mesh_outdated.on_next(True)
        # Rotation
        changed, rotation = imgui.input_float4(
            f"Rotation##{name}",
            [
                transform.rotation.w,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
            ],
        )
        if changed:
            transform.rotation = glm.quat(
                rotation[0], rotation[1], rotation[2], rotation[3]
            )
            self._mesh_outdated.on_next(True)

        # Scale
        changed, scale = imgui.drag_float3(
            f"Scale##{name}", transform.scale.to_list(), v_speed=0.01
        )
        if changed:
            transform.scale = glm.vec3(*scale)
            self._mesh_outdated.on_next(True)
        # Edit Button
        if imgui.button(f"Edit Transform##{name}"):
            if self._editing_object.value == node:
                self._editing_object.on_next(None)
            else:
                self._editing_object.on_next(node)
                self._mesh_outdated.on_next(True)
