#!/usr/bin/env python3
"""
SITL (Software-in-the-Loop) Test Runner.

Provides automated testing of the glider control system using
ArduPilot's SITL simulator.

Features:
- Automatic SITL startup and configuration
- Test scenario execution
- Telemetry recording and analysis
- Soaring behavior validation

Requirements:
- ArduPilot SITL installed (sim_vehicle.py)
- MAVProxy installed
- Python dependencies: pymavlink, dronekit

Usage:
    python run_sitl.py --scenario thermal_test
    python run_sitl.py --scenario endurance --duration 3600
    python run_sitl.py --list-scenarios
"""

import os
import sys
import time
import subprocess
import signal
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flight_control.mavlink.telemetry import TelemetryHandler, create_connection
from flight_control.mavlink.commands import FlightCommander, Waypoint

logger = logging.getLogger(__name__)


@dataclass
class SITLConfig:
    """SITL configuration."""
    vehicle: str = "ArduPlane"
    frame: str = "plane"
    speedup: int = 1
    home_lat: float = 34.0522  # Los Angeles
    home_lon: float = -118.2437
    home_alt: float = 100.0
    home_heading: float = 0.0
    model: str = "plane"
    defaults: Optional[str] = None  # Path to defaults file
    extra_args: List[str] = None

    @property
    def home_string(self) -> str:
        """Get home location string for SITL."""
        return f"{self.home_lat},{self.home_lon},{self.home_alt},{self.home_heading}"


@dataclass
class TestScenario:
    """Test scenario definition."""
    name: str
    description: str
    duration_sec: float = 300.0
    waypoints: List[Tuple[float, float, float]] = None  # (lat, lon, alt)
    initial_mode: str = "FBWA"
    test_thermal: bool = False
    thermal_location: Tuple[float, float] = None
    thermal_strength: float = 2.0
    expected_min_altitude: float = 50.0
    expected_max_altitude: float = 500.0
    success_criteria: Dict = None


# Predefined test scenarios
SCENARIOS = {
    "basic_flight": TestScenario(
        name="basic_flight",
        description="Basic flight test - takeoff, circuit, land",
        duration_sec=300,
        waypoints=[
            (34.0530, -118.2430, 150),
            (34.0540, -118.2440, 150),
            (34.0535, -118.2450, 150),
            (34.0525, -118.2440, 150),
        ],
        initial_mode="AUTO",
    ),

    "thermal_test": TestScenario(
        name="thermal_test",
        description="Test thermal detection and soaring behavior",
        duration_sec=600,
        test_thermal=True,
        thermal_location=(34.0535, -118.2435),
        thermal_strength=2.5,
        initial_mode="AUTO",
        expected_min_altitude=100,
        expected_max_altitude=400,
        success_criteria={
            "entered_thermal": True,
            "gained_altitude": 100,  # meters
        },
    ),

    "endurance": TestScenario(
        name="endurance",
        description="Extended flight for endurance testing",
        duration_sec=3600,
        waypoints=[
            (34.0530, -118.2430, 200),
            (34.0550, -118.2430, 200),
            (34.0550, -118.2450, 200),
            (34.0530, -118.2450, 200),
        ],
        initial_mode="AUTO",
    ),

    "wind_test": TestScenario(
        name="wind_test",
        description="Test behavior in windy conditions",
        duration_sec=300,
        initial_mode="FBWA",
    ),

    "rtl_test": TestScenario(
        name="rtl_test",
        description="Test Return to Launch behavior",
        duration_sec=180,
        waypoints=[
            (34.0550, -118.2430, 200),
        ],
        initial_mode="AUTO",
    ),

    "loiter_test": TestScenario(
        name="loiter_test",
        description="Test loiter/circle behavior",
        duration_sec=300,
        initial_mode="LOITER",
    ),
}


class SITLRunner:
    """
    Manages SITL simulation for testing.

    Handles starting/stopping SITL, connecting telemetry,
    and running test scenarios.
    """

    def __init__(
        self,
        config: Optional[SITLConfig] = None,
        ardupilot_path: Optional[str] = None,
        output_dir: str = "sitl_output",
    ):
        """
        Initialize SITL runner.

        Args:
            config: SITL configuration
            ardupilot_path: Path to ArduPilot source (for sim_vehicle.py)
            output_dir: Directory for logs and results
        """
        self.config = config or SITLConfig()
        self.ardupilot_path = ardupilot_path or os.environ.get(
            "ARDUPILOT_PATH",
            os.path.expanduser("~/ardupilot")
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sitl_process: Optional[subprocess.Popen] = None
        self.telemetry: Optional[TelemetryHandler] = None
        self.commander: Optional[FlightCommander] = None

        self._running = False

    def start_sitl(self, wait_ready: bool = True) -> bool:
        """
        Start SITL simulator.

        Args:
            wait_ready: Wait for SITL to be ready

        Returns:
            True if SITL started successfully
        """
        sim_vehicle = Path(self.ardupilot_path) / "Tools" / "autotest" / "sim_vehicle.py"

        if not sim_vehicle.exists():
            logger.error(f"sim_vehicle.py not found at {sim_vehicle}")
            logger.info("Set ARDUPILOT_PATH environment variable to ArduPilot source directory")
            return False

        # Build command
        cmd = [
            sys.executable,
            str(sim_vehicle),
            "-v", self.config.vehicle,
            "-f", self.config.frame,
            "--speedup", str(self.config.speedup),
            "-L", self.config.home_string,
            "--no-mavproxy",
            "--out", "udp:127.0.0.1:14550",
        ]

        if self.config.defaults:
            cmd.extend(["--defaults", self.config.defaults])

        if self.config.extra_args:
            cmd.extend(self.config.extra_args)

        logger.info(f"Starting SITL: {' '.join(cmd)}")

        try:
            self.sitl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.output_dir,
            )

            if wait_ready:
                # Wait for SITL to be ready
                logger.info("Waiting for SITL to initialize...")
                time.sleep(10)  # SITL takes time to start

                # Connect telemetry
                self.telemetry = TelemetryHandler("udp:127.0.0.1:14550")
                if self.telemetry.connect(timeout=30):
                    self.commander = FlightCommander(self.telemetry)
                    self.telemetry.start()
                    logger.info("SITL ready and connected")
                    return True
                else:
                    logger.error("Failed to connect to SITL")
                    self.stop_sitl()
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to start SITL: {e}")
            return False

    def stop_sitl(self):
        """Stop SITL simulator."""
        if self.telemetry:
            self.telemetry.stop()
            self.telemetry.disconnect()
            self.telemetry = None
            self.commander = None

        if self.sitl_process:
            logger.info("Stopping SITL...")
            self.sitl_process.terminate()
            try:
                self.sitl_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.sitl_process.kill()
            self.sitl_process = None

    def run_scenario(self, scenario: TestScenario) -> Dict:
        """
        Run a test scenario.

        Args:
            scenario: TestScenario to run

        Returns:
            Dictionary with test results
        """
        logger.info(f"Running scenario: {scenario.name}")
        logger.info(f"Description: {scenario.description}")

        results = {
            "scenario": scenario.name,
            "start_time": time.time(),
            "success": False,
            "telemetry": [],
            "events": [],
            "metrics": {},
        }

        if not self.commander:
            results["error"] = "Not connected to SITL"
            return results

        try:
            # Wait for GPS lock
            logger.info("Waiting for GPS lock...")
            start = time.time()
            while time.time() - start < 30:
                state = self.telemetry.get_state()
                if state.gps.fix_type >= 3:
                    break
                time.sleep(0.5)

            # Upload mission if waypoints provided
            if scenario.waypoints:
                waypoints = [
                    Waypoint(
                        seq=i,
                        latitude=wp[0],
                        longitude=wp[1],
                        altitude=wp[2],
                        current=1 if i == 0 else 0,
                    )
                    for i, wp in enumerate(scenario.waypoints)
                ]
                self.commander.upload_mission(waypoints)
                results["events"].append({"time": time.time(), "event": "mission_uploaded"})

            # Arm
            logger.info("Arming...")
            if self.commander.arm(force=True):
                results["events"].append({"time": time.time(), "event": "armed"})
            else:
                results["error"] = "Failed to arm"
                return results

            # Set mode
            logger.info(f"Setting mode to {scenario.initial_mode}")
            self.commander.set_mode(scenario.initial_mode)
            results["events"].append({"time": time.time(), "event": f"mode_{scenario.initial_mode}"})

            # Run test
            logger.info(f"Running test for {scenario.duration_sec}s...")
            start_time = time.time()
            sample_interval = 1.0
            last_sample = 0

            min_alt = float('inf')
            max_alt = 0
            thermal_entered = False
            altitude_gained = 0
            initial_altitude = None

            while time.time() - start_time < scenario.duration_sec:
                state = self.telemetry.get_state()

                # Record telemetry
                if time.time() - last_sample >= sample_interval:
                    results["telemetry"].append({
                        "time": time.time() - start_time,
                        "altitude": state.gps.altitude_rel,
                        "airspeed": state.airspeed.airspeed,
                        "mode": state.flight_mode_name,
                        "lat": state.gps.latitude,
                        "lon": state.gps.longitude,
                    })
                    last_sample = time.time()

                # Track altitude
                alt = state.gps.altitude_rel
                min_alt = min(min_alt, alt)
                max_alt = max(max_alt, alt)

                if initial_altitude is None and alt > 10:
                    initial_altitude = alt

                # Check for thermal mode
                if state.flight_mode_name == "THERMAL":
                    if not thermal_entered:
                        thermal_entered = True
                        results["events"].append({"time": time.time() - start_time, "event": "thermal_entered"})

                # Check for RTL test
                if scenario.name == "rtl_test" and time.time() - start_time > 60:
                    self.commander.return_to_launch()
                    results["events"].append({"time": time.time() - start_time, "event": "rtl_commanded"})

                time.sleep(0.1)

            # Calculate altitude gained
            if initial_altitude is not None:
                altitude_gained = max_alt - initial_altitude

            # Disarm
            self.commander.set_mode("FBWA")
            time.sleep(1)
            self.commander.disarm(force=True)

            # Record metrics
            results["metrics"] = {
                "min_altitude": min_alt,
                "max_altitude": max_alt,
                "altitude_gained": altitude_gained,
                "thermal_entered": thermal_entered,
                "samples_collected": len(results["telemetry"]),
            }

            # Check success criteria
            if scenario.success_criteria:
                results["success"] = True
                for key, expected in scenario.success_criteria.items():
                    actual = results["metrics"].get(key)
                    if isinstance(expected, bool):
                        if actual != expected:
                            results["success"] = False
                    elif isinstance(expected, (int, float)):
                        if actual is None or actual < expected:
                            results["success"] = False
            else:
                # Default success: completed without crash
                results["success"] = min_alt > 10

            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]

        except Exception as e:
            logger.error(f"Scenario error: {e}")
            results["error"] = str(e)

        return results

    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save test results to file."""
        if filename is None:
            filename = f"results_{results['scenario']}_{int(time.time())}.json"

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filepath}")


def list_scenarios():
    """Print available test scenarios."""
    print("\nAvailable Test Scenarios:")
    print("=" * 60)
    for name, scenario in SCENARIOS.items():
        print(f"\n{name}:")
        print(f"  Description: {scenario.description}")
        print(f"  Duration: {scenario.duration_sec}s")
        if scenario.test_thermal:
            print(f"  Thermal test: strength={scenario.thermal_strength} m/s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SITL Test Runner for HighEfficiencyGlide")
    parser.add_argument("--scenario", "-s", help="Scenario to run")
    parser.add_argument("--duration", "-d", type=float, help="Override scenario duration (seconds)")
    parser.add_argument("--list-scenarios", "-l", action="store_true", help="List available scenarios")
    parser.add_argument("--output", "-o", default="sitl_output", help="Output directory")
    parser.add_argument("--ardupilot-path", help="Path to ArduPilot source")
    parser.add_argument("--speedup", type=int, default=1, help="Simulation speedup factor")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.list_scenarios:
        list_scenarios()
        return

    if not args.scenario:
        print("Error: --scenario required")
        list_scenarios()
        return

    if args.scenario not in SCENARIOS:
        print(f"Error: Unknown scenario '{args.scenario}'")
        list_scenarios()
        return

    # Get scenario
    scenario = SCENARIOS[args.scenario]
    if args.duration:
        scenario.duration_sec = args.duration

    # Create runner
    config = SITLConfig(speedup=args.speedup)
    runner = SITLRunner(
        config=config,
        ardupilot_path=args.ardupilot_path,
        output_dir=args.output,
    )

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Interrupted, stopping...")
        runner.stop_sitl()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start SITL
        if not runner.start_sitl():
            logger.error("Failed to start SITL")
            return

        # Run scenario
        results = runner.run_scenario(scenario)

        # Save results
        runner.save_results(results)

        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Scenario: {results['scenario']}")
        print(f"Success: {results['success']}")
        print(f"Duration: {results.get('duration', 0):.1f}s")
        print("\nMetrics:")
        for key, value in results.get('metrics', {}).items():
            print(f"  {key}: {value}")

        if 'error' in results:
            print(f"\nError: {results['error']}")

    finally:
        runner.stop_sitl()


if __name__ == "__main__":
    main()
