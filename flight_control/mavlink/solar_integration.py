"""
Solar MPPT Integration Module.

Integrates solar power system data with ArduPilot telemetry:
- MPPT (Maximum Power Point Tracking) data reception
- Solar power telemetry injection into MAVLink stream
- Power management decisions based on energy state
- Battery charging optimization

The solar data can come from:
1. Custom MPPT controller via serial/I2C
2. ArduPilot's built-in solar support
3. External power monitoring system

Usage:
    solar = SolarIntegration(telemetry_handler)
    solar.connect_mppt("/dev/ttyUSB1")
    solar.start()
    state = solar.get_power_state()
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List
from datetime import datetime
from enum import IntEnum

logger = logging.getLogger(__name__)


class ChargeState(IntEnum):
    """Battery charging state."""
    UNKNOWN = 0
    DISCHARGING = 1
    CHARGING = 2
    FULL = 3
    LOW = 4
    CRITICAL = 5


class PowerMode(IntEnum):
    """Power management modes."""
    NORMAL = 0
    POWER_SAVE = 1
    HIGH_PERFORMANCE = 2
    EMERGENCY = 3


@dataclass
class MPPTData:
    """MPPT controller data."""
    panel_voltage: float = 0.0  # V
    panel_current: float = 0.0  # A
    panel_power: float = 0.0  # W
    battery_voltage: float = 0.0  # V
    battery_current: float = 0.0  # A (positive = charging)
    load_current: float = 0.0  # A
    temperature: float = 25.0  # Celsius
    efficiency: float = 0.0  # %
    pwm_duty: float = 0.0  # %
    timestamp: float = 0.0


@dataclass
class PowerState:
    """Complete power system state."""
    # Solar
    solar_power: float = 0.0  # W
    solar_voltage: float = 0.0  # V
    solar_current: float = 0.0  # A

    # Battery
    battery_voltage: float = 0.0  # V
    battery_current: float = 0.0  # A
    battery_soc: float = 100.0  # %
    battery_energy: float = 0.0  # Wh remaining
    charge_state: ChargeState = ChargeState.UNKNOWN

    # System
    load_power: float = 0.0  # W
    net_power: float = 0.0  # W (positive = surplus)
    power_mode: PowerMode = PowerMode.NORMAL
    time_remaining: float = 0.0  # hours at current consumption

    # Statistics
    energy_generated_wh: float = 0.0  # Total Wh from solar
    energy_consumed_wh: float = 0.0  # Total Wh consumed
    peak_solar_power: float = 0.0  # Peak W observed

    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "solar": {
                "power": self.solar_power,
                "voltage": self.solar_voltage,
                "current": self.solar_current,
            },
            "battery": {
                "voltage": self.battery_voltage,
                "current": self.battery_current,
                "soc": self.battery_soc,
                "charge_state": self.charge_state.name,
            },
            "system": {
                "load_power": self.load_power,
                "net_power": self.net_power,
                "power_mode": self.power_mode.name,
                "time_remaining_hours": self.time_remaining,
            },
            "statistics": {
                "energy_generated_wh": self.energy_generated_wh,
                "energy_consumed_wh": self.energy_consumed_wh,
                "peak_solar_power": self.peak_solar_power,
            },
        }


class SolarIntegration:
    """
    Solar power system integration.

    Monitors and manages solar power generation, battery charging,
    and power consumption for the glider.

    Args:
        telemetry: TelemetryHandler for MAVLink communication
        battery_capacity_wh: Battery capacity in Wh
        min_soc_percent: Minimum battery SOC to maintain
    """

    def __init__(
        self,
        telemetry=None,  # TelemetryHandler
        battery_capacity_wh: float = 50.0,
        min_soc_percent: float = 20.0,
    ):
        self.telemetry = telemetry
        self.battery_capacity_wh = battery_capacity_wh
        self.min_soc = min_soc_percent

        self.state = PowerState()
        self.mppt_data = MPPTData()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._mppt_serial = None

        # Power management callbacks
        self._mode_callbacks: List[Callable] = []

        # Integration state
        self._last_update = time.time()
        self._energy_accumulator = 0.0

    def connect_mppt(
        self,
        device: str = "/dev/ttyUSB1",
        baud: int = 9600,
        protocol: str = "victron"
    ) -> bool:
        """
        Connect to MPPT controller via serial.

        Args:
            device: Serial device path
            baud: Baud rate
            protocol: Protocol type ("victron", "epever", "custom")

        Returns:
            True if connected
        """
        try:
            import serial
            self._mppt_serial = serial.Serial(device, baud, timeout=1.0)
            self._mppt_protocol = protocol
            logger.info(f"Connected to MPPT at {device}")
            return True
        except Exception as e:
            logger.error(f"MPPT connection failed: {e}")
            return False

    def start(self):
        """Start solar monitoring thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Solar integration started")

    def stop(self):
        """Stop solar monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._mppt_serial:
            self._mppt_serial.close()
            self._mppt_serial = None
        logger.info("Solar integration stopped")

    def get_power_state(self) -> PowerState:
        """Get current power state (thread-safe)."""
        with self._lock:
            return PowerState(
                solar_power=self.state.solar_power,
                solar_voltage=self.state.solar_voltage,
                solar_current=self.state.solar_current,
                battery_voltage=self.state.battery_voltage,
                battery_current=self.state.battery_current,
                battery_soc=self.state.battery_soc,
                battery_energy=self.state.battery_energy,
                charge_state=self.state.charge_state,
                load_power=self.state.load_power,
                net_power=self.state.net_power,
                power_mode=self.state.power_mode,
                time_remaining=self.state.time_remaining,
                energy_generated_wh=self.state.energy_generated_wh,
                energy_consumed_wh=self.state.energy_consumed_wh,
                peak_solar_power=self.state.peak_solar_power,
                timestamp=self.state.timestamp,
            )

    def on_power_mode_change(self, callback: Callable):
        """Register callback for power mode changes."""
        self._mode_callbacks.append(callback)

    def set_simulated_solar(
        self,
        voltage: float,
        current: float,
        battery_voltage: float = 11.1,
    ):
        """
        Set simulated solar data (for testing without MPPT).

        Args:
            voltage: Panel voltage
            current: Panel current
            battery_voltage: Battery voltage
        """
        with self._lock:
            self.mppt_data.panel_voltage = voltage
            self.mppt_data.panel_current = current
            self.mppt_data.panel_power = voltage * current
            self.mppt_data.battery_voltage = battery_voltage
            self.mppt_data.timestamp = time.time()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Read MPPT data
                if self._mppt_serial:
                    self._read_mppt()
                else:
                    # Use simulated or telemetry data
                    self._update_from_telemetry()

                # Update power state
                self._update_state()

                # Check power mode
                self._check_power_mode()

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Solar monitor error: {e}")
                time.sleep(1.0)

    def _read_mppt(self):
        """Read data from MPPT controller."""
        if not self._mppt_serial:
            return

        try:
            # Protocol-specific parsing
            if self._mppt_protocol == "victron":
                self._parse_victron()
            elif self._mppt_protocol == "epever":
                self._parse_epever()
            else:
                self._parse_custom()

        except Exception as e:
            logger.debug(f"MPPT read error: {e}")

    def _parse_victron(self):
        """Parse Victron VE.Direct protocol."""
        # Victron sends text protocol with label/value pairs
        # Format: "LABEL\tVALUE\r\n"
        data = {}
        buffer = self._mppt_serial.read(256).decode('utf-8', errors='ignore')

        for line in buffer.split('\r\n'):
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) == 2:
                    data[parts[0]] = parts[1]

        with self._lock:
            if 'V' in data:  # Battery voltage (mV)
                self.mppt_data.battery_voltage = int(data['V']) / 1000.0
            if 'VPV' in data:  # Panel voltage (mV)
                self.mppt_data.panel_voltage = int(data['VPV']) / 1000.0
            if 'PPV' in data:  # Panel power (W)
                self.mppt_data.panel_power = int(data['PPV'])
            if 'I' in data:  # Battery current (mA)
                self.mppt_data.battery_current = int(data['I']) / 1000.0
            if 'IL' in data:  # Load current (mA)
                self.mppt_data.load_current = int(data['IL']) / 1000.0
            self.mppt_data.timestamp = time.time()

    def _parse_epever(self):
        """Parse EPEver Modbus protocol (placeholder)."""
        # EPEver uses Modbus RTU
        # Would need pymodbus library
        pass

    def _parse_custom(self):
        """Parse custom protocol."""
        # Custom JSON or CSV protocol
        line = self._mppt_serial.readline().decode('utf-8').strip()
        if line:
            try:
                import json
                data = json.loads(line)
                with self._lock:
                    self.mppt_data.panel_voltage = data.get('pv', 0)
                    self.mppt_data.panel_current = data.get('pc', 0)
                    self.mppt_data.battery_voltage = data.get('bv', 0)
                    self.mppt_data.battery_current = data.get('bc', 0)
                    self.mppt_data.timestamp = time.time()
            except json.JSONDecodeError:
                pass

    def _update_from_telemetry(self):
        """Update from ArduPilot telemetry when no MPPT connected."""
        if self.telemetry:
            telem_state = self.telemetry.get_state()
            with self._lock:
                self.mppt_data.battery_voltage = telem_state.battery.voltage
                self.mppt_data.battery_current = -telem_state.battery.current  # ArduPilot uses discharge convention
                self.mppt_data.timestamp = time.time()

    def _update_state(self):
        """Update power state from MPPT data."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        with self._lock:
            # Solar
            self.state.solar_voltage = self.mppt_data.panel_voltage
            self.state.solar_current = self.mppt_data.panel_current
            self.state.solar_power = self.mppt_data.panel_power
            if self.state.solar_power == 0 and self.state.solar_voltage > 0:
                self.state.solar_power = self.state.solar_voltage * self.state.solar_current

            # Battery
            self.state.battery_voltage = self.mppt_data.battery_voltage
            self.state.battery_current = self.mppt_data.battery_current

            # Estimate SOC from voltage (simplified)
            # For 3S LiPo: 12.6V = 100%, 9.9V = 0%
            voltage_range = 12.6 - 9.9
            self.state.battery_soc = max(0, min(100,
                (self.state.battery_voltage - 9.9) / voltage_range * 100
            ))
            self.state.battery_energy = self.battery_capacity_wh * self.state.battery_soc / 100

            # Load and net power
            self.state.load_power = self.mppt_data.load_current * self.state.battery_voltage
            if self.state.load_power == 0 and self.state.battery_current < 0:
                self.state.load_power = -self.state.battery_current * self.state.battery_voltage

            self.state.net_power = self.state.solar_power - self.state.load_power

            # Charge state
            if self.state.battery_current > 0.1:
                if self.state.battery_soc >= 99:
                    self.state.charge_state = ChargeState.FULL
                else:
                    self.state.charge_state = ChargeState.CHARGING
            elif self.state.battery_current < -0.1:
                if self.state.battery_soc < 10:
                    self.state.charge_state = ChargeState.CRITICAL
                elif self.state.battery_soc < 20:
                    self.state.charge_state = ChargeState.LOW
                else:
                    self.state.charge_state = ChargeState.DISCHARGING
            else:
                self.state.charge_state = ChargeState.UNKNOWN

            # Time remaining
            if self.state.load_power > 0 and self.state.net_power < 0:
                self.state.time_remaining = self.state.battery_energy / abs(self.state.net_power)
            else:
                self.state.time_remaining = float('inf')

            # Accumulate energy
            if dt > 0 and dt < 10:  # Sanity check
                energy_delta = self.state.solar_power * dt / 3600  # Wh
                self.state.energy_generated_wh += energy_delta
                if self.state.net_power < 0:
                    self.state.energy_consumed_wh += abs(self.state.net_power) * dt / 3600

            # Peak tracking
            if self.state.solar_power > self.state.peak_solar_power:
                self.state.peak_solar_power = self.state.solar_power

            self.state.timestamp = now

    def _check_power_mode(self):
        """Check and update power management mode."""
        with self._lock:
            old_mode = self.state.power_mode

            if self.state.charge_state == ChargeState.CRITICAL:
                self.state.power_mode = PowerMode.EMERGENCY
            elif self.state.charge_state == ChargeState.LOW:
                self.state.power_mode = PowerMode.POWER_SAVE
            elif self.state.net_power > 20:  # Good surplus
                self.state.power_mode = PowerMode.HIGH_PERFORMANCE
            else:
                self.state.power_mode = PowerMode.NORMAL

            new_mode = self.state.power_mode

        # Notify callbacks on mode change
        if new_mode != old_mode:
            logger.info(f"Power mode changed: {old_mode.name} -> {new_mode.name}")
            for callback in self._mode_callbacks:
                try:
                    callback(new_mode)
                except Exception as e:
                    logger.error(f"Power mode callback error: {e}")

    def get_recommended_actions(self) -> List[str]:
        """
        Get recommended actions based on power state.

        Returns:
            List of recommended action strings
        """
        actions = []
        state = self.get_power_state()

        if state.power_mode == PowerMode.EMERGENCY:
            actions.append("CRITICAL: Initiate RTL immediately")
            actions.append("Disable non-essential systems")
        elif state.power_mode == PowerMode.POWER_SAVE:
            actions.append("Reduce compute load (disable vision)")
            actions.append("Increase thermal soaring effort")
            actions.append("Consider reducing altitude")
        elif state.net_power < 0 and state.time_remaining < 0.5:
            actions.append("Energy deficit - seek thermals")
            actions.append("Plan route toward home")

        if state.solar_power < 10 and state.solar_voltage > 5:
            actions.append("Low solar output - check panel orientation")

        if state.charge_state == ChargeState.FULL:
            actions.append("Battery full - optimal for extended mission")

        return actions


def calculate_endurance(
    battery_wh: float,
    solar_power: float,
    consumption: float,
    min_soc: float = 20.0,
) -> Dict:
    """
    Calculate flight endurance based on power balance.

    Args:
        battery_wh: Battery capacity in Wh
        solar_power: Current solar generation in W
        consumption: Current power consumption in W
        min_soc: Minimum SOC to maintain (%)

    Returns:
        Dictionary with endurance calculations
    """
    usable_battery = battery_wh * (1 - min_soc / 100)
    net_power = solar_power - consumption

    if net_power >= 0:
        # Energy positive - unlimited endurance
        return {
            "is_sustainable": True,
            "endurance_hours": float('inf'),
            "net_power_w": net_power,
            "message": "Self-sustaining flight possible",
        }
    else:
        # Energy negative - limited by battery
        endurance = usable_battery / abs(net_power)
        return {
            "is_sustainable": False,
            "endurance_hours": endurance,
            "net_power_w": net_power,
            "message": f"Endurance limited to {endurance:.1f} hours",
        }
