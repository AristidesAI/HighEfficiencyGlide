#!/usr/bin/env python3
"""
Ground Station WebSocket Server.

Provides real-time telemetry streaming to web clients via WebSocket.
Also serves the dashboard HTML interface.

Features:
- WebSocket telemetry broadcast
- REST API for commands
- Static file serving for dashboard
- Multiple client support
- Telemetry recording

Usage:
    python server.py --mavlink udp:127.0.0.1:14550 --port 8080
    # Then open http://localhost:8080 in browser
"""

import asyncio
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Set, Optional, Dict
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flight_control.mavlink.telemetry import TelemetryHandler, TelemetryState
from flight_control.mavlink.commands import FlightCommander
from flight_control.mavlink.solar_integration import SolarIntegration

logger = logging.getLogger(__name__)


class GroundStation:
    """
    Ground station server.

    Manages telemetry connections, WebSocket clients, and command handling.
    """

    def __init__(
        self,
        mavlink_connection: str = "udp:127.0.0.1:14550",
        host: str = "0.0.0.0",
        port: int = 8080,
        static_dir: Optional[str] = None,
    ):
        if not HAS_FASTAPI:
            raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn")

        self.mavlink_connection = mavlink_connection
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).parent / "static"

        # Components
        self.telemetry: Optional[TelemetryHandler] = None
        self.commander: Optional[FlightCommander] = None
        self.solar: Optional[SolarIntegration] = None

        # WebSocket clients
        self.clients: Set[WebSocket] = set()

        # Recording
        self.recording = False
        self.record_file = None

        # FastAPI app
        self.app = FastAPI(
            title="HighEfficiencyGlide Ground Station",
            description="Real-time telemetry and control for solar glider",
            version="0.1.0",
        )
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.on_event("startup")
        async def startup():
            """Connect to MAVLink on startup."""
            await self.connect()
            asyncio.create_task(self._broadcast_loop())

        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown."""
            await self.disconnect()

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve dashboard HTML."""
            dashboard_path = self.static_dir / "dashboard.html"
            if dashboard_path.exists():
                return dashboard_path.read_text()
            else:
                # Return embedded minimal dashboard
                return self._get_embedded_dashboard()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for telemetry streaming."""
            await websocket.accept()
            self.clients.add(websocket)
            logger.info(f"Client connected. Total: {len(self.clients)}")

            try:
                while True:
                    # Receive commands from client
                    data = await websocket.receive_text()
                    await self._handle_ws_command(websocket, data)
            except WebSocketDisconnect:
                self.clients.discard(websocket)
                logger.info(f"Client disconnected. Total: {len(self.clients)}")

        @self.app.get("/api/status")
        async def get_status():
            """Get current status."""
            if self.telemetry:
                state = self.telemetry.get_state()
                return JSONResponse(state.to_dict())
            return JSONResponse({"connected": False})

        @self.app.get("/api/solar")
        async def get_solar():
            """Get solar power status."""
            if self.solar:
                state = self.solar.get_power_state()
                return JSONResponse(state.to_dict())
            return JSONResponse({"error": "Solar integration not available"})

        @self.app.post("/api/arm")
        async def arm():
            """Arm the aircraft."""
            if self.commander:
                success = self.commander.arm()
                return JSONResponse({"success": success})
            raise HTTPException(503, "Not connected")

        @self.app.post("/api/disarm")
        async def disarm():
            """Disarm the aircraft."""
            if self.commander:
                success = self.commander.disarm()
                return JSONResponse({"success": success})
            raise HTTPException(503, "Not connected")

        @self.app.post("/api/mode/{mode}")
        async def set_mode(mode: str):
            """Set flight mode."""
            if self.commander:
                success = self.commander.set_mode(mode)
                return JSONResponse({"success": success, "mode": mode})
            raise HTTPException(503, "Not connected")

        @self.app.post("/api/rtl")
        async def rtl():
            """Return to launch."""
            if self.commander:
                success = self.commander.return_to_launch()
                return JSONResponse({"success": success})
            raise HTTPException(503, "Not connected")

        @self.app.post("/api/thermal")
        async def thermal():
            """Enter thermal soaring mode."""
            if self.commander:
                success = self.commander.start_thermal()
                return JSONResponse({"success": success})
            raise HTTPException(503, "Not connected")

        @self.app.post("/api/goto")
        async def goto(lat: float, lon: float, alt: float):
            """Go to location."""
            if self.commander:
                success = self.commander.goto(lat, lon, alt)
                return JSONResponse({"success": success})
            raise HTTPException(503, "Not connected")

        @self.app.post("/api/record/start")
        async def start_recording():
            """Start telemetry recording."""
            if not self.recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.record_file = open(f"recording_{timestamp}.jsonl", "w")
                self.recording = True
                return JSONResponse({"recording": True})
            return JSONResponse({"recording": True, "message": "Already recording"})

        @self.app.post("/api/record/stop")
        async def stop_recording():
            """Stop telemetry recording."""
            if self.recording:
                self.recording = False
                if self.record_file:
                    self.record_file.close()
                    self.record_file = None
                return JSONResponse({"recording": False})
            return JSONResponse({"recording": False})

        # Mount static files if directory exists
        if self.static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

    async def connect(self):
        """Connect to MAVLink."""
        logger.info(f"Connecting to {self.mavlink_connection}")
        self.telemetry = TelemetryHandler(self.mavlink_connection)

        # Run connection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, self.telemetry.connect, 30.0)

        if connected:
            self.telemetry.start()
            self.commander = FlightCommander(self.telemetry)
            self.solar = SolarIntegration(self.telemetry)
            self.solar.start()
            logger.info("Connected to MAVLink")
        else:
            logger.error("Failed to connect to MAVLink")

    async def disconnect(self):
        """Disconnect from MAVLink."""
        if self.solar:
            self.solar.stop()
        if self.telemetry:
            self.telemetry.stop()
            self.telemetry.disconnect()
        if self.record_file:
            self.record_file.close()
        logger.info("Disconnected")

    async def _broadcast_loop(self):
        """Broadcast telemetry to all connected clients."""
        while True:
            try:
                if self.telemetry and self.clients:
                    state = self.telemetry.get_state()
                    data = {
                        "type": "telemetry",
                        "data": state.to_dict(),
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Add solar data if available
                    if self.solar:
                        data["solar"] = self.solar.get_power_state().to_dict()

                    message = json.dumps(data)

                    # Record if enabled
                    if self.recording and self.record_file:
                        self.record_file.write(message + "\n")

                    # Broadcast to all clients
                    disconnected = set()
                    for client in self.clients:
                        try:
                            await client.send_text(message)
                        except Exception:
                            disconnected.add(client)

                    self.clients -= disconnected

                await asyncio.sleep(0.1)  # 10 Hz update rate

            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1)

    async def _handle_ws_command(self, websocket: WebSocket, data: str):
        """Handle command received via WebSocket."""
        try:
            cmd = json.loads(data)
            cmd_type = cmd.get("type")
            response = {"type": "response", "command": cmd_type}

            if cmd_type == "arm":
                response["success"] = self.commander.arm() if self.commander else False
            elif cmd_type == "disarm":
                response["success"] = self.commander.disarm() if self.commander else False
            elif cmd_type == "mode":
                mode = cmd.get("mode", "FBWA")
                response["success"] = self.commander.set_mode(mode) if self.commander else False
            elif cmd_type == "rtl":
                response["success"] = self.commander.return_to_launch() if self.commander else False
            elif cmd_type == "goto":
                lat = cmd.get("lat")
                lon = cmd.get("lon")
                alt = cmd.get("alt", 100)
                response["success"] = self.commander.goto(lat, lon, alt) if self.commander else False
            else:
                response["error"] = f"Unknown command: {cmd_type}"

            await websocket.send_text(json.dumps(response))

        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
        except Exception as e:
            await websocket.send_text(json.dumps({"error": str(e)}))

    def _get_embedded_dashboard(self) -> str:
        """Return embedded minimal dashboard HTML."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>HighEfficiencyGlide Ground Station</title>
    <style>
        body { font-family: monospace; background: #1a1a2e; color: #eee; margin: 20px; }
        .panel { background: #16213e; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .title { color: #e94560; font-size: 24px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .label { color: #888; font-size: 12px; }
        .value { font-size: 24px; color: #0f3460; }
        .value.ok { color: #4ecca3; }
        .value.warn { color: #f9ed69; }
        .value.error { color: #e94560; }
        button { background: #e94560; border: none; color: white; padding: 10px 20px;
                 border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #ff6b6b; }
        button.secondary { background: #0f3460; }
        #map { height: 300px; background: #0f3460; border-radius: 8px; }
        .status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
        .status.connected { background: #4ecca3; }
        .status.disconnected { background: #e94560; }
    </style>
</head>
<body>
    <div class="title">HighEfficiencyGlide Ground Station</div>

    <div class="grid">
        <div class="panel">
            <h3><span id="conn-status" class="status disconnected"></span>Connection</h3>
            <div class="label">Mode</div>
            <div id="mode" class="value">--</div>
            <div class="label">Armed</div>
            <div id="armed" class="value">--</div>
        </div>

        <div class="panel">
            <h3>Position</h3>
            <div class="label">Altitude (m)</div>
            <div id="altitude" class="value ok">--</div>
            <div class="label">Airspeed (m/s)</div>
            <div id="airspeed" class="value">--</div>
            <div class="label">Ground Speed (m/s)</div>
            <div id="groundspeed" class="value">--</div>
        </div>

        <div class="panel">
            <h3>Attitude</h3>
            <div class="label">Roll</div>
            <div id="roll" class="value">--</div>
            <div class="label">Pitch</div>
            <div id="pitch" class="value">--</div>
            <div class="label">Heading</div>
            <div id="heading" class="value">--</div>
        </div>

        <div class="panel">
            <h3>Battery</h3>
            <div class="label">Voltage</div>
            <div id="voltage" class="value">--</div>
            <div class="label">Current</div>
            <div id="current" class="value">--</div>
            <div class="label">Remaining</div>
            <div id="remaining" class="value">--</div>
        </div>

        <div class="panel">
            <h3>Solar</h3>
            <div class="label">Power (W)</div>
            <div id="solar-power" class="value ok">--</div>
            <div class="label">Net Power (W)</div>
            <div id="net-power" class="value">--</div>
        </div>

        <div class="panel">
            <h3>Controls</h3>
            <button onclick="sendCmd('arm')">ARM</button>
            <button onclick="sendCmd('disarm')">DISARM</button>
            <br>
            <button onclick="setMode('FBWA')">FBWA</button>
            <button onclick="setMode('AUTO')">AUTO</button>
            <button onclick="setMode('LOITER')">LOITER</button>
            <button onclick="setMode('THERMAL')">THERMAL</button>
            <br>
            <button onclick="sendCmd('rtl')" class="secondary">RTL</button>
        </div>
    </div>

    <script>
        let ws;

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('conn-status').className = 'status connected';
            };

            ws.onclose = () => {
                document.getElementById('conn-status').className = 'status disconnected';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'telemetry') {
                    updateDisplay(msg.data, msg.solar);
                }
            };
        }

        function updateDisplay(data, solar) {
            document.getElementById('mode').textContent = data.flight_mode || '--';
            document.getElementById('armed').textContent = data.armed ? 'ARMED' : 'DISARMED';
            document.getElementById('armed').className = data.armed ? 'value warn' : 'value ok';

            document.getElementById('altitude').textContent = (data.gps?.altitude_rel || 0).toFixed(1);
            document.getElementById('airspeed').textContent = (data.airspeed?.airspeed || 0).toFixed(1);
            document.getElementById('groundspeed').textContent = (data.gps?.ground_speed || 0).toFixed(1);

            document.getElementById('roll').textContent = (data.attitude?.roll || 0).toFixed(1) + '°';
            document.getElementById('pitch').textContent = (data.attitude?.pitch || 0).toFixed(1) + '°';
            document.getElementById('heading').textContent = (data.gps?.heading || 0).toFixed(0) + '°';

            document.getElementById('voltage').textContent = (data.battery?.voltage || 0).toFixed(1) + 'V';
            document.getElementById('current').textContent = (data.battery?.current || 0).toFixed(1) + 'A';
            document.getElementById('remaining').textContent = (data.battery?.remaining || 0) + '%';

            if (solar) {
                document.getElementById('solar-power').textContent = (solar.solar?.power || 0).toFixed(1);
                const net = solar.system?.net_power || 0;
                document.getElementById('net-power').textContent = net.toFixed(1);
                document.getElementById('net-power').className = net > 0 ? 'value ok' : 'value warn';
            }
        }

        function sendCmd(cmd) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: cmd}));
            }
        }

        function setMode(mode) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'mode', mode: mode}));
            }
        }

        connect();
    </script>
</body>
</html>
"""

    def run(self):
        """Run the ground station server."""
        logger.info(f"Starting ground station on http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HighEfficiencyGlide Ground Station")
    parser.add_argument("--mavlink", "-m", default="udp:127.0.0.1:14550",
                        help="MAVLink connection string")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Server port")
    parser.add_argument("--static", "-s", help="Static files directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    gs = GroundStation(
        mavlink_connection=args.mavlink,
        host=args.host,
        port=args.port,
        static_dir=args.static,
    )
    gs.run()


if __name__ == "__main__":
    main()
