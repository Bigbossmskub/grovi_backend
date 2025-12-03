from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import time
import psutil
import subprocess
import os
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/tunnel", tags=["tunnel"])

# In-memory storage for tunnel data
tunnel_data = {
    "session_status": "online",
    "account": "grovi-system@example.com (Plan: Free)",
    "version": "1.0.0",
    "region": "Asia Pacific (ap)",
    "latency": "42ms",
    "web_interface": "http://127.0.0.1:4040",
    "forwarding": "https://your-tunnel.trycloudflare.com -> http://localhost:3000",
    "connections": {
        "ttl": 11,
        "opn": 0,
        "rt1": 0.08,
        "rt5": 0.03,
        "p50": 0.47,
        "p90": 5.53
    },
    "http_requests": []
}

# WebSocket connections for real-time updates
websocket_connections: List[WebSocket] = []

class TunnelManager:
    def __init__(self):
        self.request_counter = 0
        self.start_time = time.time()
    
    def generate_sample_requests(self):
        """Generate sample HTTP requests for demo"""
        sample_paths = [
            "/",
            "/api/health",
            "/api/fields",
            "/api/auth/login",
            "/api/vi-analysis",
            "/static/css/main.css",
            "/static/js/app.js",
            "/favicon.ico",
            "/api/fields/123",
            "/api/images/456"
        ]
        
        status_codes = [200, 200, 200, 200, 200, 200, 200, 404, 200, 200]
        
        for i in range(5):  # Generate 5 sample requests
            path = random.choice(sample_paths)
            status = random.choice(status_codes)
            
            request = {
                "id": self.request_counter,
                "method": "GET",
                "path": path,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "response_time": round(random.uniform(0.01, 2.0), 3),
                "size": random.randint(100, 50000)
            }
            
            tunnel_data["http_requests"].insert(0, request)
            self.request_counter += 1
            
            # Keep only last 50 requests
            if len(tunnel_data["http_requests"]) > 50:
                tunnel_data["http_requests"] = tunnel_data["http_requests"][:50]
    
    def update_connections_stats(self):
        """Update connection statistics"""
        # Simulate real-time stats updates
        tunnel_data["connections"]["ttl"] = random.randint(8, 15)
        tunnel_data["connections"]["opn"] = random.randint(0, 3)
        tunnel_data["connections"]["rt1"] = round(random.uniform(0.05, 0.15), 2)
        tunnel_data["connections"]["rt5"] = round(random.uniform(0.02, 0.08), 2)
        tunnel_data["connections"]["p50"] = round(random.uniform(0.3, 0.8), 2)
        tunnel_data["connections"]["p90"] = round(random.uniform(3.0, 8.0), 2)
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get network stats
            network = psutil.net_io_counters()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used": memory.used,
                "memory_total": memory.total,
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "uptime": time.time() - self.start_time
            }
        except Exception as e:
            print(f"Error getting system stats: {e}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_used": 0,
                "memory_total": 0,
                "bytes_sent": 0,
                "bytes_recv": 0,
                "uptime": time.time() - self.start_time
            }

tunnel_manager = TunnelManager()

@router.get("/status")
async def get_tunnel_status():
    """Get current tunnel status"""
    tunnel_manager.update_connections_stats()
    system_stats = tunnel_manager.get_system_stats()
    
    return {
        **tunnel_data,
        "system_stats": system_stats,
        "last_updated": datetime.now().isoformat()
    }

@router.get("/requests")
async def get_http_requests():
    """Get HTTP requests log"""
    return {
        "requests": tunnel_data["http_requests"],
        "total": len(tunnel_data["http_requests"])
    }

@router.post("/simulate-request")
async def simulate_http_request():
    """Simulate a new HTTP request for demo purposes"""
    tunnel_manager.generate_sample_requests()
    
    # Notify all WebSocket connections
    await broadcast_update()
    
    return {"message": "Request simulated", "total_requests": len(tunnel_data["http_requests"])}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(2)
            
            # Update stats
            tunnel_manager.update_connections_stats()
            system_stats = tunnel_manager.get_system_stats()
            
            update_data = {
                "type": "stats_update",
                "data": {
                    "connections": tunnel_data["connections"],
                    "system_stats": system_stats,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await websocket.send_text(json.dumps(update_data))
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

async def broadcast_update():
    """Broadcast update to all WebSocket connections"""
    if not websocket_connections:
        return
    
    update_data = {
        "type": "request_update",
        "data": {
            "requests": tunnel_data["http_requests"][:10],  # Send only latest 10
            "total": len(tunnel_data["http_requests"])
        }
    }
    
    disconnected = []
    for websocket in websocket_connections:
        try:
            await websocket.send_text(json.dumps(update_data))
        except:
            disconnected.append(websocket)
    
    # Remove disconnected connections
    for websocket in disconnected:
        websocket_connections.remove(websocket)

@router.get("/dashboard", response_class=HTMLResponse)
async def tunnel_dashboard():
    """Serve the tunnel dashboard HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cloudflare Tunnel Dashboard</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                background-color: #0a0a0a;
                color: #e0e0e0;
                line-height: 1.4;
                overflow-x: hidden;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid #333;
            }
            
            .header h1 {
                color: #00ff00;
                font-size: 24px;
                margin-bottom: 10px;
            }
            
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .status-card {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 20px;
            }
            
            .status-card h3 {
                color: #00ff00;
                margin-bottom: 15px;
                font-size: 16px;
            }
            
            .status-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                padding: 5px 0;
                border-bottom: 1px solid #333;
            }
            
            .status-item:last-child {
                border-bottom: none;
            }
            
            .status-label {
                color: #888;
            }
            
            .status-value {
                color: #e0e0e0;
                font-weight: bold;
            }
            
            .status-value.online {
                color: #00ff00;
            }
            
            .status-value.warning {
                color: #ffaa00;
            }
            
            .status-value.error {
                color: #ff4444;
            }
            
            .connections-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin-top: 10px;
            }
            
            .connection-item {
                text-align: center;
                padding: 10px;
                background: #2a2a2a;
                border-radius: 4px;
                border: 1px solid #444;
            }
            
            .connection-label {
                color: #888;
                font-size: 12px;
                margin-bottom: 5px;
            }
            
            .connection-value {
                color: #00ff00;
                font-size: 18px;
                font-weight: bold;
            }
            
            .requests-section {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }
            
            .requests-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .requests-header h3 {
                color: #00ff00;
            }
            
            .simulate-btn {
                background: #00ff00;
                color: #000;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                font-family: inherit;
            }
            
            .simulate-btn:hover {
                background: #00cc00;
            }
            
            .requests-list {
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #333;
                border-radius: 4px;
            }
            
            .request-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 15px;
                border-bottom: 1px solid #333;
                font-family: 'Consolas', monospace;
                font-size: 14px;
            }
            
            .request-item:last-child {
                border-bottom: none;
            }
            
            .request-method {
                color: #00ff00;
                font-weight: bold;
                width: 60px;
            }
            
            .request-path {
                color: #e0e0e0;
                flex: 1;
                margin: 0 15px;
            }
            
            .request-status {
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
                min-width: 60px;
                text-align: center;
            }
            
            .status-200 {
                background: #00ff00;
                color: #000;
            }
            
            .status-404 {
                background: #ffaa00;
                color: #000;
            }
            
            .status-500 {
                background: #ff4444;
                color: #fff;
            }
            
            .system-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            .stat-item {
                background: #2a2a2a;
                padding: 10px;
                border-radius: 4px;
                border: 1px solid #444;
                text-align: center;
            }
            
            .stat-label {
                color: #888;
                font-size: 12px;
                margin-bottom: 5px;
            }
            
            .stat-value {
                color: #00ff00;
                font-size: 16px;
                font-weight: bold;
            }
            
            .progress-bar {
                width: 100%;
                height: 4px;
                background: #333;
                border-radius: 2px;
                margin-top: 5px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background: #00ff00;
                transition: width 0.3s ease;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .pulse {
                animation: pulse 2s infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Cloudflare Tunnel Dashboard</h1>
                <p>Real-time monitoring for your tunnel connection</p>
            </div>
            
            <div class="status-grid">
                <div class="status-card">
                    <h3>Session Status</h3>
                    <div class="status-item">
                        <span class="status-label">Session Status:</span>
                        <span class="status-value online" id="session-status">online</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Account:</span>
                        <span class="status-value" id="account">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Version:</span>
                        <span class="status-value" id="version">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Region:</span>
                        <span class="status-value" id="region">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Latency:</span>
                        <span class="status-value" id="latency">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Web Interface:</span>
                        <span class="status-value" id="web-interface">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Forwarding:</span>
                        <span class="status-value" id="forwarding">Loading...</span>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>Connections</h3>
                    <div class="connections-grid">
                        <div class="connection-item">
                            <div class="connection-label">ttl</div>
                            <div class="connection-value" id="conn-ttl">0</div>
                        </div>
                        <div class="connection-item">
                            <div class="connection-label">opn</div>
                            <div class="connection-value" id="conn-opn">0</div>
                        </div>
                        <div class="connection-item">
                            <div class="connection-label">rt1</div>
                            <div class="connection-value" id="conn-rt1">0.00</div>
                        </div>
                        <div class="connection-item">
                            <div class="connection-label">rt5</div>
                            <div class="connection-value" id="conn-rt5">0.00</div>
                        </div>
                        <div class="connection-item">
                            <div class="connection-label">p50</div>
                            <div class="connection-value" id="conn-p50">0.00</div>
                        </div>
                        <div class="connection-item">
                            <div class="connection-label">p90</div>
                            <div class="connection-value" id="conn-p90">0.00</div>
                        </div>
                    </div>
                </div>
                
                <div class="status-card">
                    <h3>System Stats</h3>
                    <div class="system-stats">
                        <div class="stat-item">
                            <div class="stat-label">CPU Usage</div>
                            <div class="stat-value" id="cpu-percent">0%</div>
                            <div class="progress-bar">
                                <div class="progress-fill" id="cpu-progress" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Memory Usage</div>
                            <div class="stat-value" id="memory-percent">0%</div>
                            <div class="progress-bar">
                                <div class="progress-fill" id="memory-progress" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Uptime</div>
                            <div class="stat-value" id="uptime">0s</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Bytes Sent</div>
                            <div class="stat-value" id="bytes-sent">0</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="requests-section">
                <div class="requests-header">
                    <h3>HTTP Requests</h3>
                    <button class="simulate-btn" onclick="simulateRequest()">Simulate Request</button>
                </div>
                <div class="requests-list" id="requests-list">
                    <div class="request-item">
                        <span class="request-method">GET</span>
                        <span class="request-path">/api/health</span>
                        <span class="request-status status-200">200 OK</span>
                    </div>
                    <div class="request-item">
                        <span class="request-method">GET</span>
                        <span class="request-path">/</span>
                        <span class="request-status status-200">200 OK</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let ws = null;
            let reconnectInterval = null;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/tunnel/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('WebSocket connected');
                    if (reconnectInterval) {
                        clearInterval(reconnectInterval);
                        reconnectInterval = null;
                    }
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'stats_update') {
                        updateStats(data.data);
                    } else if (data.type === 'request_update') {
                        updateRequests(data.data.requests);
                    }
                };
                
                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    if (!reconnectInterval) {
                        reconnectInterval = setInterval(connectWebSocket, 5000);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function updateStats(data) {
                // Update connection stats
                document.getElementById('conn-ttl').textContent = data.connections.ttl;
                document.getElementById('conn-opn').textContent = data.connections.opn;
                document.getElementById('conn-rt1').textContent = data.connections.rt1.toFixed(2);
                document.getElementById('conn-rt5').textContent = data.connections.rt5.toFixed(2);
                document.getElementById('conn-p50').textContent = data.connections.p50.toFixed(2);
                document.getElementById('conn-p90').textContent = data.connections.p90.toFixed(2);
                
                // Update system stats
                if (data.system_stats) {
                    document.getElementById('cpu-percent').textContent = data.system_stats.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memory-percent').textContent = data.system_stats.memory_percent.toFixed(1) + '%';
                    document.getElementById('uptime').textContent = formatUptime(data.system_stats.uptime);
                    document.getElementById('bytes-sent').textContent = formatBytes(data.system_stats.bytes_sent);
                    
                    // Update progress bars
                    document.getElementById('cpu-progress').style.width = data.system_stats.cpu_percent + '%';
                    document.getElementById('memory-progress').style.width = data.system_stats.memory_percent + '%';
                }
            }
            
            function updateRequests(requests) {
                const requestsList = document.getElementById('requests-list');
                requestsList.innerHTML = '';
                
                requests.forEach(request => {
                    const requestItem = document.createElement('div');
                    requestItem.className = 'request-item';
                    
                    const statusClass = request.status === 200 ? 'status-200' : 
                                      request.status === 404 ? 'status-404' : 'status-500';
                    
                    requestItem.innerHTML = `
                        <span class="request-method">${request.method}</span>
                        <span class="request-path">${request.path}</span>
                        <span class="request-status ${statusClass}">${request.status} ${getStatusText(request.status)}</span>
                    `;
                    
                    requestsList.appendChild(requestItem);
                });
            }
            
            function getStatusText(status) {
                switch(status) {
                    case 200: return 'OK';
                    case 404: return 'Not Found';
                    case 500: return 'Internal Server Error';
                    default: return 'Unknown';
                }
            }
            
            function formatUptime(seconds) {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const secs = Math.floor(seconds % 60);
                return `${hours}h ${minutes}m ${secs}s`;
            }
            
            function formatBytes(bytes) {
                if (bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            async function simulateRequest() {
                try {
                    const response = await fetch('/tunnel/simulate-request', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    console.log('Request simulated:', data);
                } catch (error) {
                    console.error('Error simulating request:', error);
                }
            }
            
            async function loadInitialData() {
                try {
                    const response = await fetch('/tunnel/status');
                    const data = await response.json();
                    
                    // Update session status
                    document.getElementById('account').textContent = data.account;
                    document.getElementById('version').textContent = data.version;
                    document.getElementById('region').textContent = data.region;
                    document.getElementById('latency').textContent = data.latency;
                    document.getElementById('web-interface').textContent = data.web_interface;
                    document.getElementById('forwarding').textContent = data.forwarding;
                    
                    // Update initial stats
                    updateStats(data);
                    
                    // Update initial requests
                    if (data.http_requests) {
                        updateRequests(data.http_requests.slice(0, 10));
                    }
                    
                } catch (error) {
                    console.error('Error loading initial data:', error);
                }
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                loadInitialData();
                connectWebSocket();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
