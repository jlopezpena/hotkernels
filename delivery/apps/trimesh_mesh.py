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

# How far each vertex is moved along its radius, as a share of the radius, so
# the mesh is not a perfect sphere.
ROUGHNESS = 0.02
# Seed of the roughness and of the sampling.
SEED = 0
# Faces of an icosphere before it is subdivided; each subdivision makes four of one.
ICOSAHEDRON_FACES = 20
# By size: subdivisions of the icosphere the mesh starts from, and points
# sampled on its surface. The large mesh has 4^6 * 20 faces, the small 4^4 * 20.
MESHES = {"small": (4, 12_500), "large": (6, 200_000)}
# One line per size saying how large its data is.
SIZES = {
    size: f"{ICOSAHEDRON_FACES * 4 ** subdivisions:,} faces, {samples:,} surface samples"
    for size, (subdivisions, samples) in MESHES.items()
}


def make_data(size: str) -> tuple:
    """Build the vertices and faces of a rough sphere.

    Args:
        size: One of SIZES.

    Returns:
        (vertices as float64 rows of x, y, z, faces as int64 rows of three
        vertex indices, points to sample on the surface).
    """
    import trimesh

    subdivisions, samples = MESHES[size]
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    rng = np.random.default_rng(SEED)
    vertices = np.array(sphere.vertices) * (1 + ROUGHNESS * rng.standard_normal((len(sphere.vertices), 1)))
    return vertices, np.array(sphere.faces), samples


def run(vertices: np.ndarray, faces: np.ndarray, samples: int) -> tuple:
    """Process the mesh.

    Args:
        vertices: Vertex positions.
        faces: Vertex indices of each triangle.
        samples: Points to sample on the surface.

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
    points, sampled_faces = trimesh.sample.sample_surface(mesh, samples, seed=SEED)
    section = mesh.section(plane_origin=mesh.center_mass, plane_normal=[0.0, 0.0, 1.0])
    refined = mesh.subdivide()
    return (*properties, np.asarray(points), np.asarray(sampled_faces), np.asarray(section.vertices),
            np.asarray(refined.vertices), np.asarray(refined.faces))
