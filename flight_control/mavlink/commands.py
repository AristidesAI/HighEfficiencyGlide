"""
MAVLink Flight Commands Module.

Provides high-level commands for controlling the glider via MAVLink:
- Arm/Disarm
- Mode changes
- Waypoint navigation
- Parameter management
- Calibration commands

Usage:
    commander = FlightCommander(telemetry_handler)
    commander.set_mode("AUTO")
    commander.arm()
    commander.goto(lat, lon, alt)
"""

import time
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink2
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False

from .telemetry import TelemetryHandler, FlightMode

logger = logging.getLogger(__name__)


class CommandResult(IntEnum):
    """MAVLink command result codes."""
    ACCEPTED = 0
    TEMPORARILY_REJECTED = 1
    DENIED = 2
    UNSUPPORTED = 3
    FAILED = 4
    IN_PROGRESS = 5
    CANCELLED = 6


@dataclass
class Waypoint:
    """Mission waypoint definition."""
    seq: int
    frame: int = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
    command: int = 16  # MAV_CMD_NAV_WAYPOINT
    current: int = 0
    autocontinue: int = 1
    param1: float = 0.0  # Hold time
    param2: float = 0.0  # Acceptance radius
    param3: float = 0.0  # Pass radius
    param4: float = 0.0  # Yaw
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 100.0  # meters


class FlightCommander:
    """
    High-level flight command interface.

    Provides methods for controlling the aircraft via MAVLink commands.
    All commands include acknowledgment checking and timeout handling.

    Args:
        telemetry: TelemetryHandler instance for communication
        command_timeout: Timeout for command acknowledgments (seconds)
    """

    def __init__(
        self,
        telemetry: TelemetryHandler,
        command_timeout: float = 5.0,
    ):
        if not HAS_PYMAVLINK:
            raise ImportError("pymavlink required")

        self.telemetry = telemetry
        self.timeout = command_timeout
        self._mav = None

    @property
    def mav(self):
        """Get MAVLink connection."""
        if self.telemetry.mavlink_connection:
            return self.telemetry.mavlink_connection
        raise RuntimeError("Not connected")

    def _wait_ack(self, command_id: int) -> CommandResult:
        """
        Wait for command acknowledgment.

        Args:
            command_id: MAV_CMD ID to wait for

        Returns:
            CommandResult enum value
        """
        start = time.time()
        while time.time() - start < self.timeout:
            msg = self.mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=0.5)
            if msg and msg.command == command_id:
                return CommandResult(msg.result)
        return CommandResult.FAILED

    def arm(self, force: bool = False) -> bool:
        """
        Arm the aircraft.

        Args:
            force: Force arming (bypass checks)

        Returns:
            True if armed successfully
        """
        logger.info("Arming aircraft...")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # confirmation
            1,  # arm
            21196 if force else 0,  # force magic number
            0, 0, 0, 0, 0
        )

        result = self._wait_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        success = result == CommandResult.ACCEPTED
        logger.info(f"Arm {'successful' if success else 'failed'}: {result.name}")
        return success

    def disarm(self, force: bool = False) -> bool:
        """
        Disarm the aircraft.

        Args:
            force: Force disarming

        Returns:
            True if disarmed successfully
        """
        logger.info("Disarming aircraft...")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,  # disarm
            21196 if force else 0,
            0, 0, 0, 0, 0
        )

        result = self._wait_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        success = result == CommandResult.ACCEPTED
        logger.info(f"Disarm {'successful' if success else 'failed'}: {result.name}")
        return success

    def set_mode(self, mode: str) -> bool:
        """
        Set flight mode by name.

        Args:
            mode: Mode name (e.g., "AUTO", "LOITER", "RTL", "THERMAL")

        Returns:
            True if mode change successful
        """
        # Map mode name to number
        mode_map = {m.name: m.value for m in FlightMode}
        if mode.upper() not in mode_map:
            logger.error(f"Unknown mode: {mode}")
            return False

        mode_id = mode_map[mode.upper()]
        logger.info(f"Setting mode to {mode} ({mode_id})")

        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        # Wait for mode change confirmation
        start = time.time()
        while time.time() - start < self.timeout:
            msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
            if msg and msg.custom_mode == mode_id:
                logger.info(f"Mode changed to {mode}")
                return True
        logger.error(f"Mode change to {mode} failed")
        return False

    def goto(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        airspeed: Optional[float] = None,
    ) -> bool:
        """
        Command aircraft to fly to location.

        Uses GUIDED mode for immediate navigation.

        Args:
            latitude: Target latitude (degrees)
            longitude: Target longitude (degrees)
            altitude: Target altitude AGL (meters)
            airspeed: Optional target airspeed (m/s)

        Returns:
            True if command sent successfully
        """
        logger.info(f"Goto: lat={latitude:.6f}, lon={longitude:.6f}, alt={altitude:.1f}m")

        # Set airspeed if specified
        if airspeed:
            self.set_airspeed(airspeed)

        # Send position target
        self.mav.mav.mission_item_int_send(
            self.mav.target_system,
            self.mav.target_component,
            0,  # seq
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            2,  # current = guided mode
            1,  # autocontinue
            0, 0, 0, 0,  # params
            int(latitude * 1e7),
            int(longitude * 1e7),
            altitude
        )

        return True

    def set_airspeed(self, airspeed: float) -> bool:
        """
        Set target airspeed.

        Args:
            airspeed: Target airspeed in m/s

        Returns:
            True if command sent
        """
        logger.info(f"Setting airspeed to {airspeed:.1f} m/s")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            0,  # airspeed type
            airspeed,
            -1,  # no throttle change
            0, 0, 0, 0
        )
        return self._wait_ack(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED) == CommandResult.ACCEPTED

    def set_altitude(self, altitude: float) -> bool:
        """
        Change target altitude.

        Args:
            altitude: Target altitude AGL (meters)

        Returns:
            True if command sent
        """
        logger.info(f"Setting altitude to {altitude:.1f} m")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_ALTITUDE,
            0,
            altitude,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            0, 0, 0, 0, 0
        )
        return True

    def return_to_launch(self) -> bool:
        """
        Command Return to Launch.

        Returns:
            True if RTL mode set successfully
        """
        logger.info("Commanding RTL")
        return self.set_mode("RTL")

    def loiter_here(self, radius: float = 50.0) -> bool:
        """
        Enter loiter mode at current position.

        Args:
            radius: Loiter radius in meters

        Returns:
            True if loiter started
        """
        logger.info(f"Entering loiter with radius {radius}m")

        # Set loiter radius parameter
        self.set_parameter("WP_LOITER_RAD", radius)

        return self.set_mode("LOITER")

    def start_thermal(self) -> bool:
        """
        Enter thermal/soaring mode.

        Returns:
            True if thermal mode started
        """
        logger.info("Entering thermal soaring mode")
        return self.set_mode("THERMAL")

    def upload_mission(self, waypoints: List[Waypoint]) -> bool:
        """
        Upload mission waypoints.

        Args:
            waypoints: List of Waypoint objects

        Returns:
            True if upload successful
        """
        logger.info(f"Uploading mission with {len(waypoints)} waypoints")

        # Clear existing mission
        self.mav.mav.mission_clear_all_send(
            self.mav.target_system,
            self.mav.target_component
        )
        time.sleep(0.5)

        # Send mission count
        self.mav.mav.mission_count_send(
            self.mav.target_system,
            self.mav.target_component,
            len(waypoints)
        )

        # Wait for requests and send items
        for i, wp in enumerate(waypoints):
            # Wait for MISSION_REQUEST
            msg = self.mav.recv_match(
                type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'],
                blocking=True,
                timeout=self.timeout
            )
            if not msg or msg.seq != i:
                logger.error(f"Mission upload failed at waypoint {i}")
                return False

            # Send waypoint
            self.mav.mav.mission_item_int_send(
                self.mav.target_system,
                self.mav.target_component,
                wp.seq,
                wp.frame,
                wp.command,
                wp.current,
                wp.autocontinue,
                wp.param1,
                wp.param2,
                wp.param3,
                wp.param4,
                int(wp.latitude * 1e7),
                int(wp.longitude * 1e7),
                wp.altitude
            )

        # Wait for ACK
        msg = self.mav.recv_match(type='MISSION_ACK', blocking=True, timeout=self.timeout)
        if msg and msg.type == 0:
            logger.info("Mission upload successful")
            return True
        else:
            logger.error("Mission upload failed")
            return False

    def start_mission(self) -> bool:
        """
        Start executing uploaded mission.

        Returns:
            True if mission started
        """
        logger.info("Starting mission")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0,
            0, 0, 0, 0, 0, 0, 0
        )
        return self.set_mode("AUTO")

    def get_parameter(self, name: str) -> Optional[float]:
        """
        Get parameter value.

        Args:
            name: Parameter name

        Returns:
            Parameter value or None if not found
        """
        self.mav.mav.param_request_read_send(
            self.mav.target_system,
            self.mav.target_component,
            name.encode('utf-8'),
            -1
        )

        msg = self.mav.recv_match(type='PARAM_VALUE', blocking=True, timeout=self.timeout)
        if msg and msg.param_id == name:
            return msg.param_value
        return None

    def set_parameter(self, name: str, value: float) -> bool:
        """
        Set parameter value.

        Args:
            name: Parameter name
            value: New value

        Returns:
            True if parameter set successfully
        """
        logger.info(f"Setting {name} = {value}")

        # Get current value to determine type
        self.mav.mav.param_set_send(
            self.mav.target_system,
            self.mav.target_component,
            name.encode('utf-8'),
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )

        # Wait for confirmation
        msg = self.mav.recv_match(type='PARAM_VALUE', blocking=True, timeout=self.timeout)
        if msg and msg.param_id.rstrip('\x00') == name:
            if abs(msg.param_value - value) < 0.001:
                return True
        logger.error(f"Failed to set {name}")
        return False

    def calibrate_airspeed(self) -> bool:
        """
        Calibrate airspeed sensor (ground calibration).

        Returns:
            True if calibration started
        """
        logger.info("Starting airspeed calibration")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
            0,
            0, 0, 0, 0, 0, 1, 0  # 6th param = airspeed
        )
        return self._wait_ack(mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION) == CommandResult.ACCEPTED

    def reboot(self) -> bool:
        """
        Reboot the flight controller.

        Returns:
            True if reboot command sent
        """
        logger.warning("Rebooting flight controller")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,
            1, 0, 0, 0, 0, 0, 0  # 1 = reboot autopilot
        )
        return True

    def request_data_stream(self, stream_id: int, rate: int) -> bool:
        """
        Request specific telemetry data stream.

        Args:
            stream_id: MAV_DATA_STREAM ID
            rate: Rate in Hz

        Returns:
            True if request sent
        """
        self.mav.mav.request_data_stream_send(
            self.mav.target_system,
            self.mav.target_component,
            stream_id,
            rate,
            1  # start
        )
        return True


def create_survey_mission(
    home_lat: float,
    home_lon: float,
    survey_points: List[Tuple[float, float]],
    altitude: float = 100.0,
    loiter_radius: float = 50.0,
) -> List[Waypoint]:
    """
    Create a survey mission with waypoints.

    Args:
        home_lat: Home latitude
        home_lon: Home longitude
        survey_points: List of (lat, lon) tuples
        altitude: Flight altitude AGL
        loiter_radius: Loiter radius for hold points

    Returns:
        List of Waypoint objects for mission
    """
    waypoints = []

    # Takeoff waypoint (or launch point for glider)
    waypoints.append(Waypoint(
        seq=0,
        command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        latitude=home_lat,
        longitude=home_lon,
        altitude=altitude,
        current=1,
    ))

    # Survey waypoints
    for i, (lat, lon) in enumerate(survey_points):
        waypoints.append(Waypoint(
            seq=i + 1,
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            latitude=lat,
            longitude=lon,
            altitude=altitude,
        ))

    # Return to home
    waypoints.append(Waypoint(
        seq=len(survey_points) + 1,
        command=mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
        latitude=home_lat,
        longitude=home_lon,
        altitude=altitude,
        param3=loiter_radius,
    ))

    return waypoints
