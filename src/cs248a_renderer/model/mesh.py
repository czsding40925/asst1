from __future__ import annotations

from typing import List
from dataclasses import dataclass, field
import numpy as np
from pyglm import glm
import open3d as o3d
import slangpy as spy
from typing import Dict

from cs248a_renderer.model.primitive import Primitive
from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.scene_object import SceneObject


@dataclass
class Triangle(Primitive):
    vertices: List[glm.vec3] = field(
        default_factory=lambda: [glm.vec3(0.0) for _ in range(3)]
    )
    colors: List[glm.vec3] = field(
        default_factory=lambda: [glm.vec3(1.0, 0.0, 1.0) for _ in range(3)]
    )

    def transform(self, matrix: glm.mat4) -> Triangle:
        transformed_vertices = [
            glm.vec3(matrix * glm.vec4(v, 1.0)) for v in self.vertices
        ]
        return Triangle(vertices=transformed_vertices, colors=self.colors)

    @property
    def bounding_box(self) -> BoundingBox3D:
        min_corner = glm.vec3(np.inf)
        max_corner = glm.vec3(-np.inf)
        for v in self.vertices:
            min_corner = glm.min(min_corner, v)
            max_corner = glm.max(max_corner, v)
        return BoundingBox3D(min=min_corner, max=max_corner)

    def get_triangle(self) -> Dict:
        return {
            "vertices": [
                np.array([v.x, v.y, v.z], dtype=np.float32) for v in self.vertices
            ],
            "colors": [
                np.array([c.x, c.y, c.z], dtype=np.float32) for c in self.colors
            ],
        }


@dataclass
class Mesh(SceneObject):
    _o3d_mesh: o3d.geometry.TriangleMesh | None = None
    triangles: List[Triangle] = field(default_factory=list)
    _bounding_box: BoundingBox3D | None = None

    def __init__(self, o3d_mesh: o3d.geometry.TriangleMesh = None, **kwargs):
        super().__init__(**kwargs)
        self._o3d_mesh = o3d_mesh
        if o3d_mesh is not None:
            self.load_from_o3d(o3d_mesh)
        min = glm.vec3(np.inf)
        max = glm.vec3(-np.inf)
        for vert in self._o3d_mesh.vertices:
            v = glm.vec3(*vert)
            min = glm.min(min, v)
            max = glm.max(max, v)
        self._bounding_box = BoundingBox3D(min=min, max=max)

    def load_from_o3d(self, mesh: o3d.geometry.TriangleMesh):
        triangles = mesh.triangles
        vertices = mesh.vertices
        colors = mesh.vertex_colors

        self.triangles = []
        for tri in triangles:
            triangle = Triangle()
            for i in range(3):
                vertex_idx = tri[i]
                triangle.vertices[i] = glm.vec3(*vertices[vertex_idx])
                if len(colors) > 0:
                    triangle.colors[i] = glm.vec3(*colors[vertex_idx])
            self.triangles.append(triangle)

    @property
    def bounding_box(self) -> BoundingBox3D:
        return self._bounding_box


def create_triangle_buf(module: spy.Module, triangles: List[Triangle]) -> spy.NDBuffer:
    device = module.device
    triangle_buf = spy.NDBuffer(
        device=device,
        dtype=module.Triangle.as_struct(),
        shape=(max(len(triangles), 1),),
    )
    cursor = triangle_buf.cursor()
    for idx, triangle in enumerate(triangles):
        cursor[idx].write(triangle.get_triangle())
    cursor.apply()
    return triangle_buf
