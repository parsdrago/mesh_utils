import trimesh
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class Sphere:
    """
    A class representing a sphere
    """
    center: np.ndarray
    radius: float


def fit_to_sphere(points: np.ndarray) -> Sphere:
    """
    Fit a sphere to a set of points

    Parameters
    ----------
    points : np.ndarray
        The points to fit the sphere to

    Returns
    -------
    Sphere
        The sphere that best fits the points
    """
    center = np.zeros(3)
    radius = 0
    counts = 0

    listed_points = list(points)
    size = len(listed_points)

    for _ in range(1000):
        sampled = random.sample(listed_points, 4)
        sphere = trimesh.nsphere.fit_nsphere(sampled)

        new_count = get_points_on_sphere(points, Sphere(sphere[0], sphere[1])).shape[0]

        if new_count > counts:
            center = sphere[0]
            radius = sphere[1]
            counts = new_count

        if counts > size / 10:
            break

    return Sphere(center, radius)



def get_points_on_sphere(points: np.ndarray, sphere: Sphere) -> np.ndarray:
    """
    Get the points on the surface of a sphere

    Parameters
    ----------

    points : np.ndarray
        The points to check
    sphere : Sphere
        The sphere to check against

    Returns
    -------
    np.ndarray
        The points on the surface of the sphere
    """
    # Compute the squared distance of each point to the sphere center
    squared_distances = np.sum((points - sphere.center) ** 2, axis=1)

    # Compute the points on the sphere
    points_on_sphere = points[np.isclose(squared_distances, sphere.radius ** 2, atol=1e-6)]
    return points_on_sphere

def detect_spheres(mesh: np.ndarray, sphere_count: int) -> np.ndarray:
    """
    Detect spheres in a mesh

    Parameters
    ----------
    mesh : np.ndarray
        The mesh to detect the spheres in
    sphere_count : int
        The number of spheres to detect

    Returns
    -------
    np.ndarray
        The detected spheres
    """
    spheres = []
    mesh_counts = []

    for _ in range(sphere_count):
        sphere = fit_to_sphere(mesh)

        # Add the sphere to the list of detected spheres
        spheres.append(sphere)

        # Remove the points on the sphere surface from the mesh
        mesh_to_remove = get_points_on_sphere(mesh, sphere)
        mesh = mesh[~np.isin(mesh, mesh_to_remove).all(axis=1)]

        mesh_counts.append(mesh.shape[0])

        if mesh.shape[0] < 4:
            break

    # sort spheres by the number of points on the sphere
    spheres = [sphere for _, sphere in sorted(zip(mesh_counts, spheres), key=lambda pair: pair[0], reverse=True)]

    return spheres
