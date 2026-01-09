"""
Unit Tests for Aerodynamics Module.

Tests for:
- Atmosphere model accuracy
- Wing geometry calculations
- Glider model physics
- Known analytical solutions
"""

import pytest
import numpy as np
from typing import TYPE_CHECKING

# Import modules under test
from ..aerodynamics.atmosphere import ISAAtmosphere, get_atmosphere
from ..aerodynamics.wing_geometry import WingGeometry, create_high_ar_glider_wing, oswald_efficiency_factor
from ..aerodynamics.airfoil_data import AirfoilDatabase, get_airfoil_database, generate_naca_4digit_geometry
from ..aerodynamics.glider_model import GliderModel, MassProperties, create_default_glider


class TestAtmosphere:
    """Tests for ISA atmosphere model."""

    def setup_method(self):
        """Setup test fixtures."""
        self.atm = ISAAtmosphere()

    def test_sea_level_temperature(self):
        """Temperature at sea level should be 288.15 K."""
        T = self.atm.temperature(0)
        assert abs(T - 288.15) < 0.01

    def test_sea_level_pressure(self):
        """Pressure at sea level should be 101325 Pa."""
        P = self.atm.pressure(0)
        assert abs(P - 101325) < 1

    def test_sea_level_density(self):
        """Density at sea level should be ~1.225 kg/m³."""
        rho = self.atm.density(0)
        assert abs(rho - 1.225) < 0.001

    def test_temperature_lapse_rate(self):
        """Temperature should decrease at ~6.5 K/km in troposphere."""
        T_0 = self.atm.temperature(0)
        T_1000 = self.atm.temperature(1000)
        lapse = (T_0 - T_1000) / 1000  # K/m
        assert abs(lapse - 0.0065) < 0.0001

    def test_pressure_at_altitude(self):
        """Pressure at 5000m should be ~54000 Pa."""
        P = self.atm.pressure(5000)
        assert 50000 < P < 58000

    def test_tropopause_temperature(self):
        """Temperature at tropopause (~11km) should be ~216.65 K."""
        T = self.atm.temperature(11000)
        assert abs(T - 216.65) < 1

    def test_reynolds_number(self):
        """Reynolds number calculation should be correct."""
        # At sea level, V=15 m/s, c=0.3m
        Re = self.atm.reynolds_number(15, 0.3, 0)
        # Expected: Re = rho * V * L / mu = 1.225 * 15 * 0.3 / 1.8e-5 ≈ 306000
        assert 280000 < Re < 330000

    def test_speed_of_sound(self):
        """Speed of sound at sea level should be ~340 m/s."""
        a = self.atm.speed_of_sound(0)
        assert abs(a - 340.3) < 1

    def test_dynamic_pressure(self):
        """Dynamic pressure q = 0.5 * rho * V²."""
        q = self.atm.dynamic_pressure(20, 0)
        expected = 0.5 * 1.225 * 20**2
        assert abs(q - expected) < 1


class TestWingGeometry:
    """Tests for wing geometry calculations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.wing = WingGeometry(
            wingspan=4.0,
            root_chord=0.4,
            taper_ratio=0.5,
            sweep_angle=0.0,
            dihedral_angle=3.0,
            twist=-2.0,
        )

    def test_semispan(self):
        """Semispan should be half of wingspan."""
        assert self.wing.semispan == 2.0

    def test_tip_chord(self):
        """Tip chord = root_chord * taper_ratio."""
        assert self.wing.tip_chord == 0.2

    def test_wing_area(self):
        """Wing area for tapered wing."""
        # S = semispan * (root + tip) = 2.0 * (0.4 + 0.2) = 1.2 m²
        expected = 2.0 * (0.4 + 0.2)
        assert abs(self.wing.wing_area - expected) < 0.001

    def test_aspect_ratio(self):
        """AR = b² / S."""
        expected = 4.0**2 / self.wing.wing_area
        assert abs(self.wing.aspect_ratio - expected) < 0.01

    def test_mac_calculation(self):
        """Mean aerodynamic chord for tapered wing."""
        lam = 0.5
        expected = (2/3) * 0.4 * (1 + lam + lam**2) / (1 + lam)
        assert abs(self.wing.mean_aerodynamic_chord - expected) < 0.001

    def test_chord_at_root(self):
        """Chord at root should equal root_chord."""
        assert abs(self.wing.chord_at_span(0) - 0.4) < 0.001

    def test_chord_at_tip(self):
        """Chord at tip should equal tip_chord."""
        assert abs(self.wing.chord_at_span(self.wing.semispan) - 0.2) < 0.001

    def test_chord_linear_interpolation(self):
        """Chord should vary linearly for tapered wing."""
        mid_chord = self.wing.chord_at_span(1.0)  # At y = 1m
        expected = 0.4 * (1 - 0.5 * (1 - 0.5))  # Linear interpolation
        assert abs(mid_chord - expected) < 0.001

    def test_twist_at_tip(self):
        """Twist at tip should equal specified twist."""
        assert abs(self.wing.twist_at_span(self.wing.semispan) - (-2.0)) < 0.001

    def test_solar_panel_area(self):
        """Solar panel area calculation."""
        area = self.wing.solar_panel_area()
        # Should be positive and less than total wing area
        assert area > 0
        assert area < self.wing.wing_area

    def test_oswald_efficiency(self):
        """Oswald efficiency should be in valid range."""
        e = oswald_efficiency_factor(20, 0.4)
        assert 0.7 < e < 0.98


class TestAirfoilData:
    """Tests for airfoil database."""

    def setup_method(self):
        """Setup test fixtures."""
        self.db = get_airfoil_database()

    def test_database_not_empty(self):
        """Database should have airfoils."""
        airfoils = self.db.list_airfoils()
        assert len(airfoils) > 0

    def test_get_known_airfoil(self):
        """Should retrieve known airfoil."""
        polar = self.db.get_airfoil("E387")
        assert polar is not None
        assert polar.name == "E387"

    def test_airfoil_cl_max(self):
        """CL_max should be reasonable."""
        polar = self.db.get_airfoil("E387")
        assert 1.0 < polar.cl_max < 2.0

    def test_airfoil_cd_min(self):
        """CD_min should be small but positive."""
        polar = self.db.get_airfoil("E387")
        assert 0 < polar.cd_min < 0.02

    def test_lift_curve_slope(self):
        """Lift curve slope should be ~0.1/deg."""
        polar = self.db.get_airfoil("E387")
        assert 0.05 < polar.cl_alpha < 0.15

    def test_interpolation(self):
        """Coefficient interpolation should work."""
        polar = self.db.get_airfoil("NACA2412")
        cl_0 = polar.get_cl(0)
        cl_5 = polar.get_cl(5)
        # CL should increase with alpha
        assert cl_5 > cl_0

    def test_naca_geometry_generation(self):
        """NACA 4-digit geometry generation."""
        x, y_upper, y_lower = generate_naca_4digit_geometry("2412")
        # Should have points
        assert len(x) > 0
        # x should range from 0 to 1
        assert abs(x[0]) < 0.01
        assert abs(x[-1] - 1.0) < 0.01
        # Upper surface above lower surface
        assert np.all(y_upper >= y_lower)


class TestGliderModel:
    """Tests for glider aerodynamic model."""

    def setup_method(self):
        """Setup test fixtures."""
        self.glider = create_default_glider()

    def test_glider_creation(self):
        """Glider should be created successfully."""
        assert self.glider is not None
        assert self.glider.wing is not None
        assert self.glider.mass is not None

    def test_positive_weight(self):
        """Weight should be positive."""
        assert self.glider.mass.weight > 0

    def test_wing_loading(self):
        """Wing loading should be reasonable for glider."""
        wl = self.glider.wing_loading()
        assert 30 < wl < 150  # N/m², typical for gliders

    def test_stall_speed(self):
        """Stall speed should be reasonable."""
        vs = self.glider.stall_speed()
        assert 5 < vs < 15  # m/s, typical for gliders

    def test_lift_at_trim(self):
        """Lift should equal weight at trim."""
        from ..aerodynamics.glider_model import FlightCondition

        v = 15.0
        alt = 1000.0
        alpha = self.glider.find_trim_alpha(v, alt)
        condition = FlightCondition(velocity=v, altitude=alt, alpha=alpha)
        state = self.glider.compute_forces(condition)

        # Lift should approximately equal weight
        assert abs(state.lift - self.glider.mass.weight) / self.glider.mass.weight < 0.05

    def test_positive_glide_ratio(self):
        """Glide ratio should be positive."""
        from ..aerodynamics.glider_model import FlightCondition

        v = 15.0
        alt = 1000.0
        alpha = self.glider.find_trim_alpha(v, alt)
        condition = FlightCondition(velocity=v, altitude=alt, alpha=alpha)
        state = self.glider.compute_forces(condition)

        assert state.glide_ratio > 0

    def test_best_glide_speed(self):
        """Best glide speed should be reasonable."""
        v_bg, ld_max = self.glider.best_glide_speed()
        assert 10 < v_bg < 30  # m/s
        assert 20 < ld_max < 60  # L/D for high-efficiency glider

    def test_min_sink_speed(self):
        """Min sink speed should be less than best glide speed."""
        v_ms, sr_min = self.glider.min_sink_speed()
        v_bg, _ = self.glider.best_glide_speed()
        assert v_ms < v_bg
        assert sr_min > 0

    def test_soar_polar_k(self):
        """SOAR_POLAR_K should match formula."""
        k = self.glider.soaring_polar_k()
        expected = 16 * self.glider.mass.total_mass / self.glider.wing.wing_area
        assert abs(k - expected) < 0.1

    def test_performance_envelope(self):
        """Performance envelope should have valid data."""
        envelope = self.glider.compute_performance_envelope()

        assert "velocity" in envelope
        assert "glide_ratio" in envelope
        assert "sink_rate" in envelope

        # All glide ratios should be non-negative
        assert np.all(envelope["glide_ratio"] >= 0)


class TestPhysicsConsistency:
    """Tests for physical consistency of calculations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.glider = create_default_glider()

    def test_drag_increases_with_velocity_squared(self):
        """Drag should scale roughly with V² at constant CL."""
        from ..aerodynamics.glider_model import FlightCondition

        alt = 1000.0
        v1 = 12.0
        v2 = 18.0

        alpha1 = self.glider.find_trim_alpha(v1, alt)
        alpha2 = self.glider.find_trim_alpha(v2, alt)

        cond1 = FlightCondition(velocity=v1, altitude=alt, alpha=alpha1)
        cond2 = FlightCondition(velocity=v2, altitude=alt, alpha=alpha2)

        state1 = self.glider.compute_forces(cond1)
        state2 = self.glider.compute_forces(cond2)

        # Drag ratio should be roughly (v2/v1)² at similar CL
        drag_ratio = state2.drag / state1.drag
        v_ratio_squared = (v2 / v1) ** 2

        # Allow some variation due to CL changes
        assert 0.5 * v_ratio_squared < drag_ratio < 2.0 * v_ratio_squared

    def test_lift_proportional_to_cl(self):
        """Lift should be proportional to CL at constant q."""
        from ..aerodynamics.glider_model import FlightCondition

        v = 15.0
        alt = 1000.0

        cond1 = FlightCondition(velocity=v, altitude=alt, alpha=2.0)
        cond2 = FlightCondition(velocity=v, altitude=alt, alpha=6.0)

        state1 = self.glider.compute_forces(cond1)
        state2 = self.glider.compute_forces(cond2)

        lift_ratio = state2.lift / state1.lift
        cl_ratio = state2.cl / state1.cl

        # Should be approximately equal
        assert abs(lift_ratio - cl_ratio) / cl_ratio < 0.1

    def test_induced_drag_proportional_to_cl_squared(self):
        """Induced drag should scale with CL²."""
        from ..aerodynamics.glider_model import FlightCondition

        v = 15.0
        alt = 1000.0

        cond1 = FlightCondition(velocity=v, altitude=alt, alpha=3.0)
        cond2 = FlightCondition(velocity=v, altitude=alt, alpha=6.0)

        state1 = self.glider.compute_forces(cond1)
        state2 = self.glider.compute_forces(cond2)

        di_ratio = state2.drag_components.induced / state1.drag_components.induced
        cl_ratio_sq = (state2.cl / state1.cl) ** 2

        # Should be approximately equal
        assert abs(di_ratio - cl_ratio_sq) / cl_ratio_sq < 0.2


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
