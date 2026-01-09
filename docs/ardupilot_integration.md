# ArduPilot Integration Guide

This document describes how to integrate HighEfficiencyGlide with ArduPilot for autonomous solar glider operation.

## Overview

The ArduPilot integration provides:
- **Soaring Mode**: Autonomous thermal detection and circling
- **Energy Management**: Solar power optimization and battery monitoring
- **Telemetry**: Real-time data streaming to ground station
- **Commands**: Remote control via MAVLink

## Hardware Requirements

### Flight Controller
- Pixhawk 4 / Pixhawk 6C (recommended)
- Or compatible ArduPilot board
- Airspeed sensor (required for soaring)
- GPS with compass

### Power System
- MPPT solar charge controller
- LiPo battery (3S-4S recommended)
- Current/voltage sensor (for ArduPilot battery monitor)

### Telemetry
- 915MHz / 433MHz radio (SiK compatible)
- Or 4G/LTE modem for long range

## Parameter Configuration

### Loading Parameters

```bash
# Connect to flight controller
mavproxy.py --master=/dev/ttyUSB0 --baudrate=57600

# Load glider parameters
param load flight_control/ardupilot_config/glider_params.param

# Load soaring parameters
param load flight_control/ardupilot_config/soaring_config.param

# Write to EEPROM
param save
```

### Critical Parameters

#### Soaring Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SOAR_ENABLE` | 1 | Enable soaring controller |
| `SOAR_POLAR_K` | 280 | K = 16 × mass(kg) / area(m²) |
| `SOAR_VSPEED` | 0.7 | Min climb rate to trigger thermal (m/s) |
| `SOAR_ALT_CUTOFF` | 100 | Min altitude for soaring (m AGL) |
| `SOAR_ALT_MAX` | 400 | Max altitude in thermal (m AGL) |

**Calculating SOAR_POLAR_K:**
```python
mass = 14.0  # kg
wing_area = 0.8  # m²
SOAR_POLAR_K = 16 * mass / wing_area  # = 280
```

#### Airspeed Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ARSPD_FBW_MIN` | 1200 | Min airspeed (cm/s) = 12 m/s |
| `ARSPD_FBW_MAX` | 2500 | Max airspeed (cm/s) = 25 m/s |
| `TRIM_ARSPD_CM` | 1800 | Cruise airspeed (cm/s) = 18 m/s |

#### TECS (Total Energy Control)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TECS_SPDWEIGHT` | 2.0 | Prioritize speed over height (glider) |
| `TECS_TIME_CONST` | 5.0 | Control response time (s) |
| `TECS_SINK_MIN` | 0.5 | Expected min sink rate (m/s) |

## Soaring Operation

### How Soaring Works

1. **Detection**: Aircraft monitors vertical speed (netto variometer)
2. **Entry**: When climb rate exceeds `SOAR_VSPEED`, enters THERMAL mode
3. **Centering**: Uses Extended Kalman Filter to estimate thermal center
4. **Circling**: Banks up to `SOAR_THML_BANK` degrees to stay in lift
5. **Exit**: Leaves thermal when climb rate drops or altitude reached

### Enabling Soaring in Mission

```
# Mission Planner / QGroundControl
1. Set SOAR_ENABLE = 1
2. Create mission with waypoints above SOAR_ALT_CUTOFF
3. Set mode to AUTO
4. Aircraft will thermal opportunistically during mission
```

### Soaring Modes

| SOAR_MODE | Behavior |
|-----------|----------|
| 0 | Disabled |
| 1 | Thermal only (opportunistic during AUTO) |
| 2 | Thermal + cruise optimization |

## Ground Station

### Starting the Server

```bash
# Default connection (SITL)
python -m flight_control.ground_station.server

# Custom connection
python -m flight_control.ground_station.server \
    --mavlink udp:127.0.0.1:14550 \
    --port 8080

# Serial connection
python -m flight_control.ground_station.server \
    --mavlink /dev/ttyUSB0,57600
```

### Web Dashboard

Open `http://localhost:8080` to access:
- Real-time telemetry display
- Arm/Disarm controls
- Mode switching
- Solar power monitoring

### WebSocket API

Connect to `ws://localhost:8080/ws` for telemetry stream:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Telemetry:', data);
};

// Send command
ws.send(JSON.stringify({type: 'arm'}));
ws.send(JSON.stringify({type: 'mode', mode: 'AUTO'}));
```

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current telemetry |
| `/api/solar` | GET | Solar power status |
| `/api/arm` | POST | Arm aircraft |
| `/api/disarm` | POST | Disarm aircraft |
| `/api/mode/{mode}` | POST | Set flight mode |
| `/api/rtl` | POST | Return to Launch |
| `/api/thermal` | POST | Enter thermal mode |

## SITL Testing

### Prerequisites

```bash
# Install ArduPilot SITL
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile

# Build Plane SITL
./waf configure --board sitl
./waf plane
```

### Running Tests

```bash
# Set ArduPilot path
export ARDUPILOT_PATH=~/ardupilot

# List scenarios
python -m flight_control.sitl.run_sitl --list-scenarios

# Run thermal test
python -m flight_control.sitl.run_sitl --scenario thermal_test

# Run with speedup
python -m flight_control.sitl.run_sitl --scenario basic_flight --speedup 5
```

### Test Scenarios

| Scenario | Description | Duration |
|----------|-------------|----------|
| `basic_flight` | Takeoff, circuit, land | 5 min |
| `thermal_test` | Thermal detection and soaring | 10 min |
| `endurance` | Extended flight test | 60 min |
| `wind_test` | Wind handling | 5 min |
| `rtl_test` | Return to Launch | 3 min |

## Solar Integration

### MPPT Connection

Connect MPPT controller via serial:

```python
from flight_control.mavlink.solar_integration import SolarIntegration
from flight_control.mavlink.telemetry import TelemetryHandler

telemetry = TelemetryHandler("udp:127.0.0.1:14550")
telemetry.connect()

solar = SolarIntegration(telemetry)
solar.connect_mppt("/dev/ttyUSB1", protocol="victron")
solar.start()

# Get power state
state = solar.get_power_state()
print(f"Solar: {state.solar_power}W, Net: {state.net_power}W")
```

### Supported MPPT Protocols

| Protocol | Controller |
|----------|------------|
| `victron` | Victron SmartSolar/BlueSolar |
| `epever` | EPEver Tracer series |
| `custom` | JSON over serial |

### Power Mode Callbacks

```python
from flight_control.mavlink.solar_integration import PowerMode

def on_mode_change(mode):
    if mode == PowerMode.EMERGENCY:
        commander.return_to_launch()
    elif mode == PowerMode.POWER_SAVE:
        # Disable vision processing
        pass

solar.on_power_mode_change(on_mode_change)
```

## Telemetry Logging

### Enable Logging

```python
telemetry = TelemetryHandler(
    "udp:127.0.0.1:14550",
    log_path="logs/"
)
```

Logs saved as JSONL (JSON Lines) format:
```json
{"time": 1234567890.123, "type": "ATTITUDE", "data": {...}}
{"time": 1234567890.223, "type": "GPS_RAW_INT", "data": {...}}
```

### Analyzing Logs

```python
import json

with open("logs/telemetry_20250110.jsonl") as f:
    for line in f:
        msg = json.loads(line)
        if msg["type"] == "ATTITUDE":
            print(f"Roll: {msg['data']['roll']}")
```

## Troubleshooting

### No Heartbeat

```
Error: Heartbeat timeout
```
- Check connection string (port, baud rate)
- Verify flight controller is powered
- Check USB/serial connection

### Soaring Not Activating

1. Verify `SOAR_ENABLE = 1`
2. Check altitude > `SOAR_ALT_CUTOFF`
3. Verify airspeed sensor working
4. Check `SOAR_VSPEED` not too high

### GPS Lock Issues

- Move to open area
- Wait 3-5 minutes for cold start
- Check GPS antenna connection

## References

- [ArduPilot Soaring Documentation](https://ardupilot.org/plane/docs/soaring.html)
- [ArduSoar Paper (IROS 2018)](https://arxiv.org/abs/1802.08215)
- [MAVLink Protocol](https://mavlink.io/en/)
- [PyMAVLink Documentation](https://mavlink.io/en/mavgen_python/)
