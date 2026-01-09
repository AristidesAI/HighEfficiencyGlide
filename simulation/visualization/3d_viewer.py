"""
3D Glider Visualization.

Creates interactive 3D visualizations of the glider using PyVista.
Useful for design review and presentation.
"""

import numpy as np
from typing import Optional, Tuple, Dict

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def check_pyvista():
    """Ensure PyVista is available."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required for 3D visualization. Install with: pip install pyvista")


def create_wing_mesh(
    wing,  # WingGeometry object
    num_span_points: int = 30,
    num_chord_points: int = 15,
    include_airfoil: bool = True,
) -> "pv.PolyData":
    """
    Create 3D mesh of wing.

    Args:
        wing: WingGeometry object
        num_span_points: Points along span
        num_chord_points: Points along chord
        include_airfoil: Include airfoil thickness

    Returns:
        PyVista PolyData mesh
    """
    check_pyvista()

    # Generate spanwise stations
    y = np.linspace(0, wing.semispan, num_span_points)

    # For each station, create chord distribution
    points = []
    faces = []

    for i, yi in enumerate(y):
        chord = wing.chord_at_span(yi)
        le_x = wing.leading_edge_x(yi)
        z = wing.vertical_position(yi)
        twist = np.radians(wing.twist_at_span(yi))

        # Chord stations
        x_local = np.linspace(0, chord, num_chord_points)

        for j, xc in enumerate(x_local):
            # Airfoil thickness (NACA 4-digit approximation)
            t = 0.12  # 12% thickness
            x_norm = xc / chord

            if include_airfoil:
                # Upper surface
                yt = 5 * t * chord * (
                    0.2969 * np.sqrt(x_norm)
                    - 0.1260 * x_norm
                    - 0.3516 * x_norm**2
                    + 0.2843 * x_norm**3
                    - 0.1015 * x_norm**4
                )
            else:
                yt = 0

            # Apply twist (rotate about LE)
            x_twisted = le_x + xc * np.cos(twist) + yt * np.sin(twist)
            z_twisted = z + xc * np.sin(twist) + yt * np.cos(twist)

            points.append([x_twisted, yi, z_twisted])

            # Also add lower surface point
            if include_airfoil:
                z_lower = z + xc * np.sin(twist) - yt * np.cos(twist)
                points.append([x_twisted, yi, z_lower])

    points = np.array(points)

    # Create mesh
    cloud = pv.PolyData(points)

    # Create surface using Delaunay triangulation
    mesh = cloud.delaunay_2d()

    return mesh


def create_glider_model(
    wing,  # WingGeometry object
    fuselage_length: float = 1.5,
    fuselage_diameter: float = 0.15,
) -> "pv.PolyData":
    """
    Create complete 3D glider model.

    Args:
        wing: WingGeometry object
        fuselage_length: Fuselage length (m)
        fuselage_diameter: Fuselage diameter (m)

    Returns:
        Combined PyVista mesh
    """
    check_pyvista()

    # Create components
    meshes = []

    # Fuselage (cylinder + nose cone)
    fuselage = pv.Cylinder(
        center=(fuselage_length / 4, 0, 0),
        direction=(1, 0, 0),
        radius=fuselage_diameter / 2,
        height=fuselage_length * 0.8,
    )
    meshes.append(fuselage)

    # Nose cone
    nose = pv.Cone(
        center=(-fuselage_length * 0.3, 0, 0),
        direction=(-1, 0, 0),
        height=fuselage_length * 0.3,
        radius=fuselage_diameter / 2,
    )
    meshes.append(nose)

    # Wing (simplified as flat plate for now)
    # Right wing
    wing_points = []
    for yi in np.linspace(0, wing.semispan, 20):
        chord = wing.chord_at_span(yi)
        le_x = wing.leading_edge_x(yi)
        z = wing.vertical_position(yi)

        wing_points.append([le_x, yi, z])
        wing_points.append([le_x + chord, yi, z])

    # Left wing (mirror)
    for yi in np.linspace(0, wing.semispan, 20):
        chord = wing.chord_at_span(yi)
        le_x = wing.leading_edge_x(yi)
        z = wing.vertical_position(yi)

        wing_points.append([le_x, -yi, z])
        wing_points.append([le_x + chord, -yi, z])

    wing_cloud = pv.PolyData(np.array(wing_points))
    wing_mesh = wing_cloud.delaunay_2d()
    meshes.append(wing_mesh)

    # Horizontal tail
    tail_span = 0.4 * wing.wingspan
    tail_chord = 0.15
    tail_x = fuselage_length * 0.7
    tail_points = [
        [tail_x, -tail_span / 2, 0.05],
        [tail_x + tail_chord, -tail_span / 2, 0.05],
        [tail_x, tail_span / 2, 0.05],
        [tail_x + tail_chord, tail_span / 2, 0.05],
    ]
    tail_cloud = pv.PolyData(np.array(tail_points))
    tail_mesh = tail_cloud.delaunay_2d()
    meshes.append(tail_mesh)

    # Vertical tail
    vtail_height = 0.15
    vtail_chord = 0.12
    vtail_points = [
        [tail_x, 0, 0.05],
        [tail_x + vtail_chord, 0, 0.05],
        [tail_x + vtail_chord * 0.7, 0, 0.05 + vtail_height],
        [tail_x, 0, 0.05 + vtail_height],
    ]
    vtail_cloud = pv.PolyData(np.array(vtail_points))
    vtail_mesh = vtail_cloud.delaunay_2d()
    meshes.append(vtail_mesh)

    # Combine all meshes
    combined = meshes[0]
    for m in meshes[1:]:
        combined = combined.merge(m)

    return combined


def visualize_glider(
    wing,  # WingGeometry object
    show_solar_panels: bool = True,
    interactive: bool = True,
    screenshot_path: Optional[str] = None,
) -> None:
    """
    Interactive 3D visualization of glider.

    Args:
        wing: WingGeometry object
        show_solar_panels: Highlight solar panel area
        interactive: Show interactive window
        screenshot_path: Save screenshot to path
    """
    check_pyvista()

    # Create plotter
    plotter = pv.Plotter()

    # Add glider model
    glider = create_glider_model(wing)
    plotter.add_mesh(glider, color='lightblue', opacity=0.8, label='Glider')

    # Add solar panel overlay
    if show_solar_panels:
        solar_points = []
        for sign in [-1, 1]:
            for yi in np.linspace(0.05 * wing.semispan, 0.90 * wing.semispan, 30):
                chord = wing.chord_at_span(yi)
                le_x = wing.leading_edge_x(yi)
                z = wing.vertical_position(yi)

                # Solar panel region
                solar_le = le_x + 0.10 * chord
                solar_te = le_x + 0.70 * chord

                solar_points.append([solar_le, sign * yi, z + 0.002])
                solar_points.append([solar_te, sign * yi, z + 0.002])

        if solar_points:
            solar_cloud = pv.PolyData(np.array(solar_points))
            solar_mesh = solar_cloud.delaunay_2d()
            plotter.add_mesh(solar_mesh, color='orange', opacity=0.6, label='Solar Panels')

    # Add axes
    plotter.add_axes()

    # Add title
    plotter.add_title(
        f"Solar Glider - {wing.wingspan:.1f}m span, AR={wing.aspect_ratio:.0f}",
        font_size=12
    )

    # Camera position
    plotter.camera_position = [
        (wing.wingspan, wing.wingspan, wing.wingspan / 2),
        (0, 0, 0),
        (0, 0, 1)
    ]

    if screenshot_path:
        plotter.screenshot(screenshot_path)

    if interactive:
        plotter.show()


def create_solar_irradiance_plot(
    wing,  # WingGeometry object
    sun_elevation: float = 45.0,
    sun_azimuth: float = 180.0,
) -> "pv.PolyData":
    """
    Create wing mesh colored by solar irradiance.

    Args:
        wing: WingGeometry object
        sun_elevation: Sun elevation angle (degrees)
        sun_azimuth: Sun azimuth from north (degrees)

    Returns:
        PyVista mesh with irradiance scalars
    """
    check_pyvista()

    # Create basic wing mesh
    points = []
    irradiance = []

    sun_vec = np.array([
        np.cos(np.radians(sun_elevation)) * np.sin(np.radians(sun_azimuth)),
        np.cos(np.radians(sun_elevation)) * np.cos(np.radians(sun_azimuth)),
        np.sin(np.radians(sun_elevation))
    ])

    for sign in [-1, 1]:
        for yi in np.linspace(0, wing.semispan, 30):
            chord = wing.chord_at_span(yi)
            le_x = wing.leading_edge_x(yi)
            z = wing.vertical_position(yi)
            dihedral = np.radians(wing.dihedral_angle)

            for xc in np.linspace(0, chord, 15):
                x = le_x + xc
                y = sign * yi

                points.append([x, y, z])

                # Surface normal (accounting for dihedral)
                normal = np.array([0, sign * np.sin(dihedral), np.cos(dihedral)])

                # Irradiance = max(0, dot(normal, sun_vec))
                irr = max(0, np.dot(normal, sun_vec))
                irradiance.append(irr)

    mesh = pv.PolyData(np.array(points))
    mesh['irradiance'] = np.array(irradiance)
    mesh = mesh.delaunay_2d()

    return mesh


def animate_sun_path(
    wing,  # WingGeometry object
    latitude: float = 40.0,
    output_path: Optional[str] = None,
) -> None:
    """
    Create animation of solar irradiance over a day.

    Args:
        wing: WingGeometry object
        latitude: Latitude for sun path
        output_path: Path for output GIF/video
    """
    check_pyvista()

    plotter = pv.Plotter(off_screen=output_path is not None)

    # Time steps (hours)
    hours = np.linspace(6, 18, 25)  # 6 AM to 6 PM

    def update_frame(hour):
        plotter.clear()

        # Calculate sun position
        hour_angle = 15 * (hour - 12)  # degrees
        declination = 23.45  # Summer solstice

        sin_elev = (
            np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) *
            np.cos(np.radians(hour_angle))
        )
        elevation = np.degrees(np.arcsin(max(-1, min(1, sin_elev))))

        cos_az = (
            np.sin(np.radians(declination)) -
            np.sin(np.radians(latitude)) * sin_elev
        ) / (np.cos(np.radians(latitude)) * np.cos(np.radians(np.arcsin(sin_elev))))
        azimuth = np.degrees(np.arccos(max(-1, min(1, cos_az))))
        if hour > 12:
            azimuth = 360 - azimuth

        # Create irradiance mesh
        mesh = create_solar_irradiance_plot(wing, elevation, azimuth)
        plotter.add_mesh(mesh, scalars='irradiance', cmap='hot', clim=[0, 1])
        plotter.add_title(f"Hour: {hour:.1f}, Sun Elevation: {elevation:.0f}°")

    if output_path:
        plotter.open_gif(output_path)
        for hour in hours:
            update_frame(hour)
            plotter.write_frame()
        plotter.close()
    else:
        # Interactive
        for hour in hours:
            update_frame(hour)
            plotter.show(auto_close=False)


class GliderViewer:
    """
    Interactive glider viewer with controls.

    Provides GUI controls for adjusting view, showing/hiding
    components, and animating.
    """

    def __init__(self, wing):
        """Initialize viewer with wing geometry."""
        check_pyvista()
        self.wing = wing
        self.plotter = None

    def show(self, background: str = 'white'):
        """
        Display interactive viewer.

        Args:
            background: Background color
        """
        self.plotter = pv.Plotter()
        self.plotter.set_background(background)

        # Add glider
        glider = create_glider_model(self.wing)
        self.plotter.add_mesh(glider, color='lightgray', opacity=0.9)

        # Add solar panels
        solar_area = self.wing.solar_panel_area()

        # Info text
        info = f"""
        Wingspan: {self.wing.wingspan:.2f} m
        Wing Area: {self.wing.wing_area:.3f} m²
        Aspect Ratio: {self.wing.aspect_ratio:.1f}
        Solar Area: {solar_area:.3f} m²
        """
        self.plotter.add_text(info, position='upper_left', font_size=10)

        # Add coordinate axes
        self.plotter.add_axes()

        # Show
        self.plotter.show()

    def export_stl(self, filepath: str):
        """Export glider model to STL file."""
        glider = create_glider_model(self.wing)
        glider.save(filepath)
