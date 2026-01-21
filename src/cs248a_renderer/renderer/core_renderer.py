"""
Core rendering module
"""

import slangpy as spy
from typing import Tuple, List, Dict
from pyglm import glm
import numpy as np
from reactivex.subject import BehaviorSubject

from cs248a_renderer import RendererModules
from cs248a_renderer.model.scene import Scene
from cs248a_renderer.model.mesh import Triangle, create_triangle_buf
from cs248a_renderer.model.volumes import create_volume_buf
from cs248a_renderer.model.bvh import BVH, create_bvh_node_buf


class Renderer:
    _device: spy.Device
    _render_target: spy.Texture

    sqrt_spp: int = 1

    # Primitive buffers.
    _triangle_buf: spy.NDBuffer | None
    _triangle_count: int | None
    _volume_tex_buf: spy.NDBuffer | None
    _volume_buf: spy.NDBuffer | None
    _volume_count: int | None
    _bvh_node_buf: spy.NDBuffer | None
    _use_bvh: bool = False
    _max_nodes: int = 0

    _sphere_sdf_buf: spy.NDBuffer | None
    _sphere_sdf_count: int | None

    _custom_sdf: Dict
    _render_custom_sdf: bool = False

    def __init__(
        self,
        device: spy.Device,
        render_texture_sbj: BehaviorSubject[Tuple[spy.Texture, int]] | None = None,
        render_texture: spy.Texture | None = None,
        render_modules: RendererModules | None = None,
    ) -> None:
        self._device = device

        def update_render_target(texture: Tuple[spy.Texture, int]):
            self._render_target = texture[0]

        if render_texture is not None:
            self._render_target = render_texture
        elif render_texture_sbj is not None:
            render_texture_sbj.subscribe(update_render_target)
        else:
            raise ValueError(
                "Must provide a render_texture or render_texture_sbj for VolumeRenderer."
            )

        # Load renderer module.
        if render_modules is None:
            render_modules = RendererModules(device=device)
        self.primitive_module = render_modules.primitive_module
        self.texture_module = render_modules.texture_module
        self.model_module = render_modules.model_module
        self.renderer_module = render_modules.renderer_module

        # Initialize primitive buffers.
        self._triangle_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.Triangle.as_struct(), shape=(1,)
        )
        self._triangle_count = 0
        self._volume_tex_buf = spy.NDBuffer(
            device=device, dtype=self.texture_module.float4.as_struct(), shape=(1,)
        )
        self._volume_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.Volume.as_struct(), shape=(1,)
        )
        self._volume_count = 0
        self._bvh_node_buf = spy.NDBuffer(
            device=device, dtype=self.model_module.BVHNode.as_struct(), shape=(1,)
        )
        self._max_nodes = 0
        self._sphere_sdf_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.SphereSDF.as_struct(), shape=(1,)
        )
        self._sphere_sdf_count = 0
        self._cube_sdf_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.CubeSDF.as_struct(), shape=(1,)
        )
        self._cube_sdf_count = 0
        self._custom_sdf = {
            "cubeSize": [1.0, 1.0, 1.0],
            "sphereRadius": 0.5,
            "invModelMatrix": np.identity(4, dtype=np.float32),
        }

    def load_triangles(self, scene: Scene) -> None:
        """Load a scene into the renderer."""
        triangles = scene.extract_triangles()
        self._triangle_buf = create_triangle_buf(self.primitive_module, triangles)
        self._triangle_count = len(triangles)
        # Clear BVH when loading new triangles.
        self._bvh_node_buf = spy.NDBuffer(
            device=self._device, dtype=self.model_module.BVHNode.as_struct(), shape=(1,)
        )
        self._max_nodes = 0
        self._use_bvh = False

    def load_volumes(self, scene: Scene) -> None:
        """Load volumes into the renderer."""
        volumes = scene.extract_volumes()
        self._volume_buf, self._volume_tex_buf = create_volume_buf(
            self.primitive_module, volumes
        )
        self._volume_count = len(volumes)

    def load_bvh(self, triangles: List[Triangle], bvh: BVH) -> None:
        self._triangle_buf = create_triangle_buf(self.primitive_module, triangles)
        self._triangle_count = len(triangles)
        self._bvh_node_buf = create_bvh_node_buf(self.model_module, bvh.nodes)
        self._max_nodes = len(bvh.nodes)
        self._use_bvh = True

    def load_sdf_spheres(self, sphere_buffer: spy.NDBuffer, sphere_count: int) -> None:
        """Load SDF spheres into the renderer."""
        self._sphere_sdf_buf = sphere_buffer
        self._sphere_sdf_count = sphere_count

    def load_sdf_cubes(self, cube_buffer: spy.NDBuffer, cube_count: int) -> None:
        """Load SDF cubes into the renderer."""
        self._cube_sdf_buf = cube_buffer
        self._cube_sdf_count = cube_count

    def set_custom_sdf(self, custom_sdf: Dict, render_custom_sdf: bool = False) -> None:
        """Load custom SDF into the renderer."""
        self._custom_sdf = custom_sdf
        self._render_custom_sdf = render_custom_sdf

    def render(
        self,
        view_mat: glm.mat4,
        fov: float,
        render_depth: bool = False,
        render_normal: bool = False,
    ) -> None:
        """Render the loaded scene."""
        focal_length = (0.5 * float(self._render_target.height)) / np.tan(
            np.radians(fov) / 2.0
        )
        uniforms = {
            "camera": {
                "invViewMatrix": np.ascontiguousarray(
                    glm.inverse(view_mat), dtype=np.float32
                ),
                "canvasSize": [
                    self._render_target.width,
                    self._render_target.height,
                ],
                "focalLength": float(focal_length),
            },
            "ambientColor": [0.0, 0.0, 0.0, 1.0],
            "sqrtSpp": self.sqrt_spp,
            "triangleCount": self._triangle_count,
            "volumeCount": self._volume_count,
            "useBVH": self._use_bvh,
            "renderDepth": render_depth,
            "renderNormal": render_normal,
        }
        if self._triangle_buf is not None:
            uniforms["triangleBuf"] = self._triangle_buf
        if self._volume_tex_buf is not None:
            uniforms["volumeTexBuf"] = {
                "buffer": self._volume_tex_buf,
            }
        if self._volume_buf is not None:
            uniforms["volumeBuf"] = self._volume_buf
        if self._bvh_node_buf is not None:
            uniforms["bvh"] = {
                "nodes": self._bvh_node_buf,
                "maxNodes": self._max_nodes,
                "primitives": self._triangle_buf,
                "numPrimitives": self._triangle_count,
            }
        sdf_uniforms = {
            "sphereCount": self._sphere_sdf_count,
            "cubeCount": self._cube_sdf_count,
            "customSDF": self._custom_sdf,
            "renderCustomSDF": self._render_custom_sdf,
        }
        if self._sphere_sdf_buf is not None:
            sdf_uniforms["spheres"] = self._sphere_sdf_buf
        if self._cube_sdf_buf is not None:
            sdf_uniforms["cubes"] = self._cube_sdf_buf
        uniforms["sdfBuf"] = sdf_uniforms
        self.renderer_module.render(
            tid=spy.grid(shape=(self._render_target.height, self._render_target.width)),
            uniforms=uniforms,
            _result=self._render_target,
        )
