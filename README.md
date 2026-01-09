# HighEfficiencyGlide

Solar-powered autonomous glider simulation and control system.

## Overview

This project provides tools for designing, simulating, and optimizing solar-powered gliders with:
- Aerodynamic simulation and wing optimization
- Solar cell placement and energy balance modeling
- ArduPilot integration for autonomous flight
- YOLO-based vision systems for GPS-denied navigation

## Project Structure

```
HighEfficiencyGlide/
├── simulation/              # Core simulation modules
│   ├── aerodynamics/        # Atmosphere, airfoils, wing geometry, glider model
│   ├── optimization/        # Genetic algorithm, objectives, constraints
│   ├── solar/               # Solar cell modeling, placement, energy balance
│   ├── visualization/       # Plots and 3D viewer
│   └── tests/               # Unit tests and scenarios
├── flight_control/          # ArduPilot integration
│   ├── ardupilot_config/    # Parameter files (.param)
│   ├── mavlink/             # Telemetry, commands, solar integration
│   ├── ground_station/      # WebSocket server and dashboard
│   └── sitl/                # Software-in-the-loop testing
├── vision/                  # YOLO vision systems
│   ├── osd_reader/          # OSD telemetry extraction
│   ├── navigation/          # GPS-denied navigation
│   └── flybywire/           # Vision-enhanced control
├── hardware/                # Hardware configurations
│   ├── raspberry_pi/
│   ├── jetson_nano/
│   └── arduino/
└── docs/                    # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HighEfficiencyGlide.git
cd HighEfficiencyGlide

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Quick Start

### Run Aerodynamic Analysis

```bash
python -m simulation.main analyze --wingspan 4.0 --ar 20
```

### Generate Performance Envelope

```bash
python -m simulation.main envelope --altitude 1000 --output plots/
```

### Run Wing Optimization

```bash
python -m simulation.main optimize --generations 50 --population 100
```

### Start Ground Station

```bash
# For SITL testing
python -m flight_control.ground_station.server --mavlink udp:127.0.0.1:14550

# For real hardware
python -m flight_control.ground_station.server --mavlink /dev/ttyUSB0,57600
```

Then open http://localhost:8080 in your browser.

## Key Features

### Aerodynamic Simulation
- ISA atmosphere model (0-11km altitude)
- NACA and high-performance airfoil database (E387, S1223, HQW2512)
- Parametric wing geometry with taper, sweep, twist, dihedral
- Vortex-induced drag modeling with Oswald efficiency

### Wing Optimization
- NSGA-II multi-objective genetic algorithm
- Objectives: Maximize L/D, maximize solar area
- Constraints: Stability margin, wing loading, weight, stall speed
- Pareto front visualization

### Solar Energy Modeling
- IV curve modeling with temperature/angle effects
- Optimal cell placement on wing surface
- Energy balance simulation over daily cycle
- MPPT controller integration

### Test Scenarios
- Calm, steady wind, turbulence, gusts
- Thermal soaring conditions
- Monte Carlo robustness analysis

## Target Specifications

| Parameter | Target |
|-----------|--------|
| Wingspan | 3-5 m |
| Weight | 10-20 kg |
| Glide Ratio | 40:1+ |
| Solar Power | 100-150 W |
| Endurance | Self-sustaining |

## ArduPilot Integration

### Soaring Parameters

The critical parameter for soaring is `SOAR_POLAR_K`:

```
SOAR_POLAR_K = 16 × mass(kg) / wing_area(m²)
```

For a 14kg glider with 0.8m² wing: `K = 16 × 14 / 0.8 = 280`

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SOAR_ENABLE` | 1 | Enable autonomous soaring |
| `SOAR_POLAR_K` | 280 | Glider polar coefficient |
| `SOAR_VSPEED` | 0.7 | Min climb for thermal (m/s) |
| `SOAR_ALT_CUTOFF` | 100 | Min altitude for soaring (m) |
| `TECS_SPDWEIGHT` | 2.0 | Speed priority in glide |

### Loading Parameters

```bash
# Connect via MAVProxy
mavproxy.py --master=/dev/ttyUSB0 --baudrate=57600

# Load configurations
param load flight_control/ardupilot_config/glider_params.param
param load flight_control/ardupilot_config/soaring_config.param
param save
```

### Ground Station Features

- **Real-time telemetry**: Altitude, airspeed, attitude, GPS
- **Solar monitoring**: Panel power, battery state, net power balance
- **Flight controls**: Arm/Disarm, mode switching, RTL
- **WebSocket API**: Stream telemetry to custom applications
- **Recording**: Log telemetry to JSONL files

### SITL Testing

```bash
# Set ArduPilot path
export ARDUPILOT_PATH=~/ardupilot

# List available scenarios
python -m flight_control.sitl.run_sitl --list-scenarios

# Run thermal soaring test
python -m flight_control.sitl.run_sitl --scenario thermal_test

# Run with speedup
python -m flight_control.sitl.run_sitl --scenario basic_flight --speedup 5
```

### MAVLink API

```python
from flight_control.mavlink.telemetry import TelemetryHandler
from flight_control.mavlink.commands import FlightCommander

# Connect
telemetry = TelemetryHandler("udp:127.0.0.1:14550")
telemetry.connect()
telemetry.start()

# Get state
state = telemetry.get_state()
print(f"Altitude: {state.gps.altitude_rel}m")
print(f"Airspeed: {state.airspeed.airspeed} m/s")

# Send commands
commander = FlightCommander(telemetry)
commander.arm()
commander.set_mode("AUTO")
commander.start_thermal()
```

## Dependencies

### Core
- Python 3.9+
- NumPy, SciPy, Matplotlib, PyYAML

### Visualization
- PyVista, VTK

### ArduPilot
- PyMAVLink >= 2.4.0
- Dronekit >= 2.9.2

### Ground Station
- FastAPI, Uvicorn, WebSockets

### Vision (optional)
- Ultralytics (YOLO)
- OpenCV
- PyTorch

## Documentation

- [ArduPilot Integration Guide](docs/ardupilot_integration.md) - Detailed setup and configuration
- [Simulation API](simulation/) - Aerodynamic modeling documentation
- [ArduPilot Soaring](https://ardupilot.org/plane/docs/soaring.html) - Official ArduPilot docs

## License

MIT License

## References

- [ArduPilot Soaring Documentation](https://ardupilot.org/plane/docs/soaring.html)
- [ArduSoar: Open-Source Thermalling Controller (IROS 2018)](https://arxiv.org/abs/1802.08215)
- [ERAU Aerospace Flight Vehicles](https://eaglepubs.erau.edu/introductiontoaerospaceflightvehicles/)
- [12-Hour Solar Flight with ArduPilot](https://discuss.ardupilot.org/t/continuous-12-hour-solar-and-soaring-flight-with-ardupilot/60235)
