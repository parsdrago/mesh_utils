import numpy as np
from detect_sphere import Sphere,fit_to_sphere, get_points_on_sphere, detect_spheres


def generate_sphere_surface_points(n_points: int, radius: float = 1, center: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """
    Generate points on the surface of a sphere

    Parameters
    ----------
    n_points : int
        Number of points to generate
    radius : float
        Radius of the sphere
    center : array-like
        Center of the sphere

    Returns
    -------
    np.ndarray
        Points on the surface of the sphere
    """
    # Generate points on the surface of a sphere
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]
    return np.array([x, y, z]).T


def generate_hemi_sphere_surface_points(n_points: int, radius: float = 1, center: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """
    Generate points on the surface of a hemisphere

    Parameters
    ----------
    n_points : int
        Number of points to generate
    radius : float
        Radius of the hemisphere
    center : array-like
        Center of the hemisphere

    Returns
    -------
    np.ndarray
        Points on the surface of the hemisphere
    """
    # Generate points on the surface of a hemisphere
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi/2, n_points)
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]
    return np.array([x, y, z]).T



def test_detect_single_sphere():
    # Generate a single sphere
    n_points = 1000
    sphere = generate_sphere_surface_points(n_points)

    # Compute the bounding sphere of the mesh
    sphere_fit = fit_to_sphere(sphere)

    # Check that the center of the sphere is close to the origin

    assert np.allclose(sphere_fit.center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(sphere_fit.radius, 1, atol=0.001)


def test_detect_single_sphere_offset():
    # Generate a single sphere
    n_points = 1000
    center = np.array([1, 2, 3])
    sphere = generate_sphere_surface_points(n_points, center=center)

    # Compute the bounding sphere of the mesh
    sphere_fit = fit_to_sphere(sphere)

    # Check that the center of the sphere is close to the origin
    assert np.allclose(sphere_fit.center, center, atol=0.001)
    assert np.allclose(sphere_fit.radius, 1, atol=0.001)


def test_detect_single_sphere_from_hemi_sphere():
    # Generate a single hemisphere
    n_points = 1000
    sphere = generate_sphere_surface_points(n_points)

    # Compute the bounding sphere of the mesh
    sphere_fit = fit_to_sphere(sphere)

    # Check that the center of the sphere is close to the origin
    assert np.allclose(sphere_fit.center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(sphere_fit.radius, 1, atol=0.001)


def test_get_points_on_sphere():
    # Generate a single sphere
    n_points = 1000
    sphere = generate_sphere_surface_points(n_points, radius=1, center=np.array([0, 0, 0]))

    n_dummy_points = 100
    dummy_points = np.random.uniform(-10, 10, (n_dummy_points, 3))

    # Combine the sphere and dummy points
    points = np.vstack([sphere, dummy_points])

    # Compute the bounding sphere of the mesh
    sphere = Sphere(center=np.array([0, 0, 0]), radius=1)

    # Compute the points on the sphere
    points_on_sphere = get_points_on_sphere(points, sphere)

    # Check that all points on the sphere are on the surface of the sphere
    assert len(points_on_sphere) == n_points

    # Check that all points on the sphere are on the surface of the sphere
    distances = np.linalg.norm(points_on_sphere - sphere.center, axis=1)
    assert np.allclose(distances, sphere.radius, atol=0.001)


def test_detect_spheres():
    # Generate a single sphere
    n_points1 = 1000
    sphere1 = generate_sphere_surface_points(n_points1, radius=1, center=np.array([0, 0, 0]))

    # Generate a second sphere
    n_points2 = 300
    sphere2 = generate_sphere_surface_points(n_points2, radius=2, center=np.array([1, 1, 1]))

    # Combine the spheres
    points = np.vstack([sphere1, sphere2])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([1, 1, 1]), atol=0.001)
    assert np.allclose(spheres[1].radius, 2, atol=0.001)


def test_detect_spheres_in_dummy():
    # Generate a single sphere
    n_points1 = 1000
    sphere1 = generate_sphere_surface_points(n_points1, radius=1, center=np.array([0, 0, 0]))

    # Generate a second sphere
    n_points2 = 300
    sphere2 = generate_sphere_surface_points(n_points2, radius=2, center=np.array([1, 1, 1]))

    n_points3 = 300
    dummy_points = np.random.uniform(-3, 3, (n_points3, 3))

    # Combine the spheres
    points = np.vstack([sphere1, sphere2, dummy_points])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([1, 1, 1]), atol=0.001)
    assert np.allclose(spheres[1].radius, 2, atol=0.001)


def test_detect_sphere_from_hemi_sphere():
    # Generate a single hemisphere
    n_points = 1000
    sphere = generate_hemi_sphere_surface_points(n_points)

    # Compute the bounding sphere of the mesh
    sphere_fit = fit_to_sphere(sphere)

    # Check that the center of the sphere is close to the origin
    assert np.allclose(sphere_fit.center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(sphere_fit.radius, 1, atol=0.001)


def test_detect_sphere_from_hemi_sphere_offset():
    # Generate a single hemisphere
    n_points = 1000
    center = np.array([1, 2, 3])
    sphere = generate_hemi_sphere_surface_points(n_points, center=center)

    # Compute the bounding sphere of the mesh
    sphere_fit = fit_to_sphere(sphere)

    # Check that the center of the sphere is close to the origin
    assert np.allclose(sphere_fit.center, center, atol=0.001)
    assert np.allclose(sphere_fit.radius, 1, atol=0.001)


def test_detect_two_spheres_from_hemi_sphere():
    # Generate a single hemisphere
    n_points1 = 1000
    sphere1 = generate_hemi_sphere_surface_points(n_points1)

    # Generate a second hemisphere
    n_points2 = 300
    sphere2 = generate_hemi_sphere_surface_points(n_points2, center=np.array([1, 1, 1]))

    # Combine the spheres
    points = np.vstack([sphere1, sphere2])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([1, 1, 1]), atol=0.001)
    assert np.allclose(spheres[1].radius, 1, atol=0.001)


def test_detect_two_spheres_with_different_radiuses_from_hemi_sphere():
    # Generate a single hemisphere
    n_points1 = 1000
    sphere1 = generate_hemi_sphere_surface_points(n_points1)

    # Generate a second hemisphere
    n_points2 = 300
    sphere2 = generate_hemi_sphere_surface_points(n_points2, radius=2)

    # Combine the spheres
    points = np.vstack([sphere1, sphere2])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[1].radius, 2, atol=0.001)


def test_detect_two_spheres_with_different_radiuses_from_hemi_sphere_with_dummy_points():
    # Generate a single hemisphere
    n_points1 = 1000
    sphere1 = generate_hemi_sphere_surface_points(n_points1)

    # Generate a second hemisphere
    n_points2 = 300
    sphere2 = generate_hemi_sphere_surface_points(n_points2, radius=2)

    n_points3 = 500
    dummy_points = np.random.uniform(-3, 3, (n_points3, 3))

    # Combine the spheres
    points = np.vstack([sphere1, sphere2, dummy_points])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[1].radius, 2, atol=0.001)


def test_detect_two_spheres_with_different_radiuses_from_hemi_sphere_offset_with_dummy_points():
    # Generate a single hemisphere
    n_points1 = 1000
    sphere1 = generate_hemi_sphere_surface_points(n_points1, center=np.array([0, 0, 0]))

    # Generate a second hemisphere
    n_points2 = 300
    sphere2 = generate_hemi_sphere_surface_points(n_points2, radius=2, center=np.array([1, 0, 0]))

    n_points3 = 500
    dummy_points = np.random.uniform(-3, 3, (n_points3, 3))

    # Combine the spheres
    points = np.vstack([sphere1, sphere2, dummy_points])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([1, 0, 0]), atol=0.001)
    assert np.allclose(spheres[1].radius, 2, atol=0.001)


def test_detect_two_spheres_with_different_radiuses_from_hemi_sphere_offset_with_dummy_points_with_large_example():
    # Generate a single hemisphere
    n_points1 = 20000
    sphere1 = generate_hemi_sphere_surface_points(n_points1, center=np.array([0, 0, 0]))

    # Generate a second hemispher
    n_points2 = 10000
    sphere2 = generate_hemi_sphere_surface_points(n_points2, radius=2, center=np.array([1, 0, 0]))

    n_points3 = 10000
    dummy_points = np.random.uniform(-3, 3, (n_points3, 3))

    # Combine the spheres
    points = np.vstack([sphere1, sphere2, dummy_points])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([1, 0, 0]), atol=0.001)
    assert np.allclose(spheres[1].radius, 2, atol=0.001)


def test_detect_two_spheres_with_different_radiuses_from_hemi_sphere_offset_with_dummy_points_with_large_example_with_unrounded():
    # Generate a single hemisphere
    n_points1 = 20000
    sphere1 = generate_hemi_sphere_surface_points(n_points1, radius=1.414, center=np.array([0, 0, 0]))

    # Generate a second hemispher
    n_points2 = 10000
    sphere2 = generate_hemi_sphere_surface_points(n_points2, radius=3.141, center=np.array([0.123, 0, 0]))

    n_points3 = 10000
    dummy_points = np.random.uniform(-3, 3, (n_points3, 3))

    # Combine the spheres
    points = np.vstack([sphere1, sphere2, dummy_points])

    # Detect the spheres
    spheres = detect_spheres(points, sphere_count=2)

    # Check that the detected spheres are close to the original spheres
    assert len(spheres) == 2
    assert np.allclose(spheres[0].center, np.array([0, 0, 0]), atol=0.001)
    assert np.allclose(spheres[0].radius, 1.414, atol=0.001)
    assert np.allclose(spheres[1].center, np.array([0.123, 0, 0]), atol=0.001)
    assert np.allclose(spheres[1].radius, 3.141, atol=0.001)
