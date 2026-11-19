"""Processes a triangle mesh with trimesh.

What a script that receives a mesh does first: build it, which merges
duplicate vertices and drops degenerate faces, read its geometric properties,
sample points on its surface, cut it with a plane, and refine it. Every step
is a documented function or attribute of the library.
"""
from __future__ import annotations

import numpy as np

TITLE = "Processing a triangle mesh: build, measure, sample, section, subdivide"
SOURCE = "trimesh, called as a library"
LICENCE = None
REQUIRES = ("trimesh",)

# Subdivisions of the icosphere the mesh starts from: 4^6 * 20 faces.
SUBDIVISIONS = 6
# How far each vertex is moved along its radius, as a share of the radius, so
# the mesh is not a perfect sphere.
ROUGHNESS = 0.02
# Points sampled on the surface.
SAMPLES = 200_000
# Seed of the roughness and of the sampling.
SEED = 0
SIZE = f"{20 * 4 ** SUBDIVISIONS:,} faces, {SAMPLES:,} surface samples"


def make_data() -> tuple:
    """Build the vertices and faces of a rough sphere.

    Returns:
        (vertices as float64 rows of x, y, z, faces as int64 rows of three
        vertex indices).
    """
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=SUBDIVISIONS)
    rng = np.random.default_rng(SEED)
    vertices = np.array(sphere.vertices) * (1 + ROUGHNESS * rng.standard_normal((len(sphere.vertices), 1)))
    return vertices, np.array(sphere.faces)


def run(vertices: np.ndarray, faces: np.ndarray) -> tuple:
    """Process the mesh.

    Args:
        vertices: Vertex positions.
        faces: Vertex indices of each triangle.

    Returns:
        The properties read, the sampled points and their faces, the vertices
        of the plane section and the refined mesh.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=True)
    properties = (
        mesh.area, mesh.volume, mesh.center_mass, mesh.moment_inertia, mesh.is_watertight, mesh.euler_number,
        mesh.face_adjacency, mesh.face_adjacency_angles, mesh.vertex_normals, mesh.edges_unique, mesh.vertex_defects,
    )
    points, sampled_faces = trimesh.sample.sample_surface(mesh, SAMPLES, seed=SEED)
    section = mesh.section(plane_origin=mesh.center_mass, plane_normal=[0.0, 0.0, 1.0])
    refined = mesh.subdivide()
    return (*properties, np.asarray(points), np.asarray(sampled_faces), np.asarray(section.vertices),
            np.asarray(refined.vertices), np.asarray(refined.faces))
