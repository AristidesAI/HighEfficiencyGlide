"""
MAVLink Telemetry Handler.

Handles real-time telemetry data from ArduPilot via MAVLink protocol.
Provides parsed data for ground station display and logging.

Features:
- Connection management (serial, UDP, TCP)
- Message parsing and caching
- Telemetry rate control
- Data logging to file
- Event callbacks for state changes

Usage:
    handler = TelemetryHandler("udp:127.0.0.1:14550")
    handler.connect()
    handler.on_message("HEARTBEAT", callback_fn)
    handler.start()
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Any
from datetime import datetime
from pathlib import Path
from enum import IntEnum
import json

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False

logger = logging.getLogger(__name__)


class FlightMode(IntEnum):
    """ArduPilot Plane flight modes."""
    MANUAL = 0
    CIRCLE = 1
    STABILIZE = 2
    TRAINING = 3
    ACRO = 4
    FBWA = 5
    FBWB = 6
    CRUISE = 7
    AUTOTUNE = 8
    AUTO = 10
    RTL = 11
    LOITER = 12
    TAKEOFF = 13
    AVOID_ADSB = 14
    GUIDED = 15
    INITIALISING = 16
    QSTABILIZE = 17
    QHOVER = 18
    QLOITER = 19
    QLAND = 20
    QRTL = 21
    QAUTOTUNE = 22
    QACRO = 23
    THERMAL = 24


@dataclass
class GPSState:
    """GPS fix information."""
    fix_type: int = 0  # 0=No fix, 2=2D, 3=3D
    satellites: int = 0
    latitude: float = 0.0  # degrees
    longitude: float = 0.0  # degrees
    altitude_msl: float = 0.0  # meters
    altitude_rel: float = 0.0  # meters AGL
    ground_speed: float = 0.0  # m/s
    heading: float = 0.0  # degrees
    hdop: float = 99.9
    vdop: float = 99.9
    timestamp: float = 0.0


@dataclass
class AttitudeState:
    """Aircraft attitude."""
    roll: float = 0.0  # degrees
    pitch: float = 0.0  # degrees
    yaw: float = 0.0  # degrees
    roll_rate: float = 0.0  # deg/s
    pitch_rate: float = 0.0  # deg/s
    yaw_rate: float = 0.0  # deg/s
    timestamp: float = 0.0


@dataclass
class AirspeedState:
    """Airspeed information."""
    airspeed: float = 0.0  # m/s (indicated)
    true_airspeed: float = 0.0  # m/s
    temperature: float = 20.0  # Celsius
    pressure: float = 101325.0  # Pa
    timestamp: float = 0.0


@dataclass
class BatteryState:
    """Battery status."""
    voltage: float = 0.0  # Volts
    current: float = 0.0  # Amps
    remaining: int = 100  # Percent
    consumed: float = 0.0  # mAh
    timestamp: float = 0.0


@dataclass
class SoaringState:
    """Thermal soaring state."""
    is_thermalling: bool = False
    thermal_strength: float = 0.0  # m/s
    thermal_radius: float = 0.0  # m
    thermal_x: float = 0.0  # m from aircraft
    thermal_y: float = 0.0  # m from aircraft
    netto_vario: float = 0.0  # m/s
    sink_rate: float = 0.0  # m/s
    timestamp: float = 0.0


@dataclass
class TelemetryState:
    """Complete telemetry state."""
    connected: bool = False
    armed: bool = False
    flight_mode: FlightMode = FlightMode.MANUAL
    flight_mode_name: str = "UNKNOWN"

    gps: GPSState = field(default_factory=GPSState)
    attitude: AttitudeState = field(default_factory=AttitudeState)
    airspeed: AirspeedState = field(default_factory=AirspeedState)
    battery: BatteryState = field(default_factory=BatteryState)
    soaring: SoaringState = field(default_factory=SoaringState)

    system_status: int = 0
    uptime_sec: float = 0.0
    last_heartbeat: float = 0.0
    message_rate: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "connected": self.connected,
            "armed": self.armed,
            "flight_mode": self.flight_mode_name,
            "gps": {
                "fix_type": self.gps.fix_type,
                "satellites": self.gps.satellites,
                "latitude": self.gps.latitude,
                "longitude": self.gps.longitude,
                "altitude_msl": self.gps.altitude_msl,
                "altitude_rel": self.gps.altitude_rel,
                "ground_speed": self.gps.ground_speed,
                "heading": self.gps.heading,
            },
            "attitude": {
                "roll": self.attitude.roll,
                "pitch": self.attitude.pitch,
                "yaw": self.attitude.yaw,
            },
            "airspeed": {
                "airspeed": self.airspeed.airspeed,
                "true_airspeed": self.airspeed.true_airspeed,
            },
            "battery": {
                "voltage": self.battery.voltage,
                "current": self.battery.current,
                "remaining": self.battery.remaining,
            },
            "soaring": {
                "is_thermalling": self.soaring.is_thermalling,
                "thermal_strength": self.soaring.thermal_strength,
                "netto_vario": self.soaring.netto_vario,
            },
            "uptime_sec": self.uptime_sec,
        }


class TelemetryHandler:
    """
    MAVLink telemetry handler.

    Connects to ArduPilot and provides parsed telemetry data.

    Args:
        connection_string: MAVLink connection string
            - Serial: "/dev/ttyUSB0" or "COM3"
            - UDP: "udp:127.0.0.1:14550"
            - TCP: "tcp:127.0.0.1:5760"
        source_system: MAVLink system ID for this handler
        log_path: Path to save telemetry logs (optional)
    """

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        source_system: int = 255,
        log_path: Optional[str] = None,
    ):
        if not HAS_PYMAVLINK:
            raise ImportError("pymavlink is required. Install with: pip install pymavlink")

        self.connection_string = connection_string
        self.source_system = source_system
        self.log_path = Path(log_path) if log_path else None

        self.mavlink_connection = None
        self.state = TelemetryState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Message callbacks
        self._callbacks: Dict[str, List[Callable]] = {}
        self._message_count = 0
        self._last_rate_check = time.time()

        # Logging
        self._log_file = None
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)

    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to MAVLink endpoint.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to {self.connection_string}")

            self.mavlink_connection = mavutil.mavlink_connection(
                self.connection_string,
                source_system=self.source_system,
                baud=57600,
            )

            # Wait for heartbeat
            logger.info("Waiting for heartbeat...")
            msg = self.mavlink_connection.wait_heartbeat(timeout=timeout)

            if msg:
                with self._lock:
                    self.state.connected = True
                    self.state.last_heartbeat = time.time()
                logger.info(f"Connected to system {self.mavlink_connection.target_system}")
                return True
            else:
                logger.error("Heartbeat timeout")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from MAVLink endpoint."""
        self.stop()
        if self.mavlink_connection:
            self.mavlink_connection.close()
            self.mavlink_connection = None
        with self._lock:
            self.state.connected = False
        logger.info("Disconnected")

    def start(self):
        """Start telemetry reception thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info("Telemetry handler started")

        # Start logging if configured
        if self.log_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_path / f"telemetry_{timestamp}.jsonl"
            self._log_file = open(log_file, "w")
            logger.info(f"Logging to {log_file}")

    def stop(self):
        """Stop telemetry reception."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        logger.info("Telemetry handler stopped")

    def on_message(self, message_type: str, callback: Callable):
        """
        Register callback for specific message type.

        Args:
            message_type: MAVLink message name (e.g., "HEARTBEAT", "ATTITUDE")
            callback: Function to call with message data
        """
        if message_type not in self._callbacks:
            self._callbacks[message_type] = []
        self._callbacks[message_type].append(callback)

    def get_state(self) -> TelemetryState:
        """Get current telemetry state (thread-safe copy)."""
        with self._lock:
            # Return a shallow copy - state objects are replaced, not modified
            return TelemetryState(
                connected=self.state.connected,
                armed=self.state.armed,
                flight_mode=self.state.flight_mode,
                flight_mode_name=self.state.flight_mode_name,
                gps=self.state.gps,
                attitude=self.state.attitude,
                airspeed=self.state.airspeed,
                battery=self.state.battery,
                soaring=self.state.soaring,
                system_status=self.state.system_status,
                uptime_sec=self.state.uptime_sec,
                last_heartbeat=self.state.last_heartbeat,
                message_rate=self.state.message_rate,
            )

    def _receive_loop(self):
        """Main telemetry reception loop."""
        while self._running:
            try:
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=1.0)
                if msg:
                    self._process_message(msg)
                    self._message_count += 1

                # Update message rate every second
                now = time.time()
                if now - self._last_rate_check >= 1.0:
                    with self._lock:
                        self.state.message_rate = self._message_count / (now - self._last_rate_check)
                    self._message_count = 0
                    self._last_rate_check = now

                # Check for heartbeat timeout
                with self._lock:
                    if now - self.state.last_heartbeat > 5.0:
                        self.state.connected = False

            except Exception as e:
                logger.error(f"Receive error: {e}")
                time.sleep(0.1)

    def _process_message(self, msg):
        """Process incoming MAVLink message."""
        msg_type = msg.get_type()
        now = time.time()

        with self._lock:
            if msg_type == "HEARTBEAT":
                self.state.connected = True
                self.state.last_heartbeat = now
                self.state.armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                self.state.system_status = msg.system_status

                # Decode flight mode
                if msg.type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
                    try:
                        self.state.flight_mode = FlightMode(msg.custom_mode)
                        self.state.flight_mode_name = self.state.flight_mode.name
                    except ValueError:
                        self.state.flight_mode_name = f"MODE_{msg.custom_mode}"

            elif msg_type == "GLOBAL_POSITION_INT":
                self.state.gps.latitude = msg.lat / 1e7
                self.state.gps.longitude = msg.lon / 1e7
                self.state.gps.altitude_msl = msg.alt / 1000.0
                self.state.gps.altitude_rel = msg.relative_alt / 1000.0
                self.state.gps.heading = msg.hdg / 100.0
                self.state.gps.ground_speed = (msg.vx**2 + msg.vy**2)**0.5 / 100.0
                self.state.gps.timestamp = now

            elif msg_type == "GPS_RAW_INT":
                self.state.gps.fix_type = msg.fix_type
                self.state.gps.satellites = msg.satellites_visible
                self.state.gps.hdop = msg.eph / 100.0 if msg.eph < 65535 else 99.9
                self.state.gps.vdop = msg.epv / 100.0 if msg.epv < 65535 else 99.9

            elif msg_type == "ATTITUDE":
                import math
                self.state.attitude.roll = math.degrees(msg.roll)
                self.state.attitude.pitch = math.degrees(msg.pitch)
                self.state.attitude.yaw = math.degrees(msg.yaw)
                self.state.attitude.roll_rate = math.degrees(msg.rollspeed)
                self.state.attitude.pitch_rate = math.degrees(msg.pitchspeed)
                self.state.attitude.yaw_rate = math.degrees(msg.yawspeed)
                self.state.attitude.timestamp = now

            elif msg_type == "VFR_HUD":
                self.state.airspeed.airspeed = msg.airspeed
                self.state.airspeed.true_airspeed = msg.groundspeed  # Approximation
                self.state.airspeed.timestamp = now

            elif msg_type == "SYS_STATUS":
                self.state.battery.voltage = msg.voltage_battery / 1000.0
                self.state.battery.current = msg.current_battery / 100.0
                self.state.battery.remaining = msg.battery_remaining
                self.state.battery.timestamp = now

            elif msg_type == "BATTERY_STATUS":
                if msg.current_consumed >= 0:
                    self.state.battery.consumed = msg.current_consumed

            elif msg_type == "SYSTEM_TIME":
                self.state.uptime_sec = msg.time_boot_ms / 1000.0

        # Trigger callbacks
        if msg_type in self._callbacks:
            for callback in self._callbacks[msg_type]:
                try:
                    callback(msg)
                except Exception as e:
                    logger.error(f"Callback error for {msg_type}: {e}")

        # Log message
        if self._log_file:
            try:
                log_entry = {
                    "time": now,
                    "type": msg_type,
                    "data": msg.to_dict() if hasattr(msg, 'to_dict') else str(msg),
                }
                self._log_file.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass


def create_connection(
    connection_type: str = "udp",
    host: str = "127.0.0.1",
    port: int = 14550,
    device: str = "/dev/ttyUSB0",
    baud: int = 57600,
) -> str:
    """
    Create MAVLink connection string.

    Args:
        connection_type: "udp", "tcp", or "serial"
        host: Host address for UDP/TCP
        port: Port for UDP/TCP
        device: Serial device path
        baud: Serial baud rate

    Returns:
        Connection string for TelemetryHandler
    """
    if connection_type == "udp":
        return f"udp:{host}:{port}"
    elif connection_type == "tcp":
        return f"tcp:{host}:{port}"
    elif connection_type == "serial":
        return f"{device},{baud}"
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")
