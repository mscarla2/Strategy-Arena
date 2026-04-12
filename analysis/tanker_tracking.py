import asyncio
import websockets
import json
import os
import sys
import csv
from datetime import datetime, timezone

# --- Configuration (Wider box to catch more traffic) ---
HORMUZ_STRAIT = {
    "name": "Strait of Hormuz",
    "min_lat": 24.0, # Widened from 25.5
    "max_lat": 28.0, # Widened from 27.0
    "min_lon": 53.0, # Widened from 55.5
    "max_lon": 60.0  # Widened from 58.5
}

# AISStream format: [[[lat1, lon1], [lat2, lon2]]]
AIS_STREAM_BBOX = [[HORMUZ_STRAIT["min_lat"], HORMUZ_STRAIT["min_lon"]], 
                   [HORMUZ_STRAIT["max_lat"], HORMUZ_STRAIT["max_lon"]]]

TANKER_TYPES = [80, 81, 82, 83, 84, 89]

class HormuzMonitor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.vessels = {}
        self.logs = [] # To show live activity
        self.total_msgs = 0
        self.written_to_csv = set()  # Track which tankers have been saved

    def add_log(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {text}")
        if len(self.logs) > 5: self.logs.pop(0)

    def get_direction(self, course):
        if course is None: return "Unknown"
        return "Outbound" if 90 <= course <= 210 else "Inbound" if 270 <= course <= 360 else "Transit"

    def write_tanker_to_csv(self, mmsi, data):
        """Write a single tanker record to ship_info.csv"""
        file_exists = os.path.isfile("ship_info.csv")
        with open("ship_info.csv", "a", newline='', encoding='utf-8') as f:
            fieldnames = ["MMSI", "Name", "Size (m)", "Direction", "Laden", "Destination", "Entry Time (UTC)"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        self.written_to_csv.add(mmsi)

    def analyze_current_state(self):
        now = datetime.now(timezone.utc)
        tanker_details = []
        
        # 1. Filter for vessels we have seen recently (last 30 mins)
        active_vessels = [v for v in self.vessels.values() 
                        if (now - v['last_seen']).total_seconds() < 1800]
        
        # 2. To be a "Tanker" in the census, it must have:
        #    - is_tanker == True (from a Static Message)
        #    - lat/lon data (from a Position Message)
        tankers = [v for v in active_vessels if v.get('is_tanker') is True and v.get('lat') is not None]
        
        vlccs = [v for v in tankers if v.get('length', 0) > 300]

        for t in tankers:
            tanker_details.append({
                "mmsi": t['mmsi'],
                "name": t.get('name', 'IDENTIFYING...'),
                "size": f"{t.get('length', '???')}m" if t.get('length') else "---",
                "direction": self.get_direction(t.get('course')),
                "laden": (t.get('draught') or 0) > 15.0,
                "dest": t.get('dest', '---')
            })

        stats = {
            "timestamp": now.strftime('%Y-%m-%d %H:%M:%S UTC'),
            "bbox": f"{HORMUZ_STRAIT['min_lat']}N/{HORMUZ_STRAIT['min_lon']}E",
            "total_vessels": len(active_vessels),
            "total_tankers": len(tankers),
            "vlcc_count": len(vlccs),
            "laden_count": sum(1 for t in tanker_details if t['laden']),
            "outbound_count": sum(1 for t in tanker_details if t['direction'] == "Outbound"),
            "inbound_count": sum(1 for t in tanker_details if t['direction'] == "Inbound"),
            "tanker_details": tanker_details
        }
        
        disruption = {
            "expected_vlcc_in_bbox": 4.0,
            "observed_vlcc": stats["vlcc_count"],
            "vlcc_vs_baseline_pct": (stats["vlcc_count"] / 4.0) * 100,
            "expected_tankers_in_bbox": 12.0,
            "observed_tankers": stats["total_tankers"],
            "tanker_vs_baseline_pct": (stats["total_tankers"] / 12.0) * 100,
            "severity": "NORMAL" if stats["vlcc_count"] >= 3 else "MODERATE" if stats["vlcc_count"] >= 1 else "CRITICAL",
            "color": "GREEN" if stats["vlcc_count"] >= 3 else "ORANGE" if stats["vlcc_count"] >= 1 else "RED",
            "wti_signal": "Stable flow." if stats["vlcc_count"] >= 3 else "Supply tightness.",
            "caveat": "Free live stream. Tankers show as 'IDENTIFYING' until static data arrives."
        }
        return stats, disruption

    async def run_monitor(self):
        url = "wss://stream.aisstream.io/v0/stream"
        
        async with websockets.connect(url) as websocket:
            subscribe_msg = {
                "APIKey": self.api_key,
                "BoundingBoxes": [AIS_STREAM_BBOX] # No message type filter = more data!
            }
            await websocket.send(json.dumps(subscribe_msg))
            self.add_log("Subscription sent. Monitoring Strait...")

            last_ui_update = 0
            async for message in websocket:
                self.total_msgs += 1
                msg = json.loads(message)
                mmsi = msg.get("MetaData", {}).get("MMSI")
                ship_name = msg.get("MetaData", {}).get("ShipName", "").strip()
                
                # New vessel: create entry and log entry message
                if mmsi not in self.vessels:
                    entry_time = datetime.now(timezone.utc)
                    self.vessels[mmsi] = {
                        'mmsi': mmsi,
                        'last_seen': entry_time,
                        'entry_time': entry_time,
                        'is_tanker': False,
                        'entry_logged': False   # for entry notification
                    }
                
                v = self.vessels[mmsi]
                v['last_seen'] = datetime.now(timezone.utc)
                if ship_name:
                    v['name'] = ship_name

                # Log entry if not already done (use best available name)
                if not v.get('entry_logged', False):
                    name_for_log = ship_name if ship_name else str(mmsi)
                    self.add_log(f"SHIP entered Hormuz: {name_for_log}")
                    v['entry_logged'] = True

                # Process message types
                m_type = msg["MessageType"]
                if m_type == "PositionReport":
                    pos = msg["Message"]["PositionReport"]
                    v.update({
                        "lat": pos.get("Latitude"),
                        "lon": pos.get("Longitude"),
                        "course": pos.get("Cog"),
                        "draught": pos.get("Draught")
                    })
                elif m_type == "ShipStaticData":
                    static = msg["Message"]["ShipStaticData"]
                    ship_type = static.get("ShipType")
                    is_tanker = ship_type in TANKER_TYPES
                    length = static.get("Dimension", {}).get("A", 0) + static.get("Dimension", {}).get("B", 0)
                    v.update({
                        "is_tanker": is_tanker,
                        "length": length,
                        "dest": static.get("Destination", "").strip(),
                        "name": static.get("Name", "").strip() or v.get("name", "")
                    })
                    if is_tanker:
                        self.add_log(f"ID'd Tanker: {v.get('name', mmsi)}")

                # If this is a tanker and we haven't written it to CSV yet, do it now
                if v.get('is_tanker') and mmsi not in self.written_to_csv:
                    # Prepare CSV row with current data
                    row = {
                        "MMSI": mmsi,
                        "Name": v.get('name', ''),
                        "Size (m)": v.get('length', ''),
                        "Direction": self.get_direction(v.get('course')),
                        "Laden": "Yes" if (v.get('draught') or 0) > 15.0 else "No" if v.get('draught') else "",
                        "Destination": v.get('dest', ''),
                        "Entry Time (UTC)": v['entry_time'].strftime('%Y-%m-%d %H:%M:%S UTC')
                    }
                    self.write_tanker_to_csv(mmsi, row)
                    self.add_log(f"Saved tanker {v.get('name', mmsi)} to ship_info.csv")

                # Update screen every 2 seconds
                if asyncio.get_event_loop().time() - last_ui_update > 2:
                    stats, disruption = self.analyze_current_state()
                    self.refresh_screen(stats, disruption)
                    last_ui_update = asyncio.get_event_loop().time()

    def refresh_screen(self, stats, disruption):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear' if os.environ.get('TERM') else 'echo ""')
        print_snapshot_report(stats, disruption, "AISStream.io (Free Live)")
        print(f"\n  LIVE ACTIVITY LOG (Total Messages: {self.total_msgs})")
        print(f"  {'─' * 62}")
        for log in self.logs:
            print(f"  {log}")

def print_snapshot_report(stats: dict, disruption: dict, source: str):
    # --- EXACTLY YOUR ORIGINAL FORMAT ---
    W = 66
    print(f"\n{'═' * W}")
    print(f"  STRAIT OF HORMUZ — TANKER TRAFFIC SNAPSHOT")
    print(f"  Source : {source}")
    print(f"  Time   : {stats['timestamp']}")
    print(f"  BBox   : {stats['bbox']}")
    print(f"{'═' * W}")

    print(f"\n  VESSEL CENSUS")
    print(f"  {'─' * 40}")
    print(f"  Total vessels detected:       {stats['total_vessels']:>5d}")
    print(f"  Tankers identified:           {stats['total_tankers']:>5d}")
    print(f"  VLCCs (300m+):                {stats['vlcc_count']:>5d}")
    print(f"  Laden tankers (est.):         {stats['laden_count']:>5d}")
    print(f"  Outbound (Exiting):           {stats['outbound_count']:>5d}")
    print(f"  Inbound  (Entering):          {stats['inbound_count']:>5d}")

    print(f"\n  DISRUPTION ANALYSIS vs. NORMAL BASELINE")
    print(f"  {'─' * 40}")
    print(f"  Expected VLCCs (normal): {disruption['expected_vlcc_in_bbox']:>5.1f} | Observed: {disruption['observed_vlcc']:>5d}")
    print(f"  Expected Tankers (normal): {disruption['expected_tankers_in_bbox']:>5.1f} | Observed: {disruption['observed_tankers']:>5d}")

    sev, col, wti = disruption["severity"], disruption["color"], disruption["wti_signal"]
    print(f"\n  ┌{'─' * (W - 4)}┐")
    print(f"  │  SEVERITY : {sev:<{W-18}}│")
    print(f"  │  SIGNAL   : {col:<{W-18}}│")
    print(f"  │  WTI IMPLICATION: {wti:<{W-24}}│")
    print(f"  └{'─' * (W - 4)}┘")

    print(f"\n  TANKER DETAIL (TOP 5)")
    print(f"  {'─' * (W - 2)}")
    if stats["tanker_details"]:
        for v in stats["tanker_details"][:5]:
            laden = "Yes" if v["laden"] else "No"
            print(f"  {str(v['mmsi']):<12} {v['name'][:18]:<18} {v['size']:<8} {v['direction']:<10} {laden:<5} {v['dest'][:10]}")
    else:
        print("  Scanning for Tanker Static Data (can take up to 6 mins)...")

if __name__ == "__main__":
    API_KEY = "658cefb4934340ae7cd43b9e7c3fc3b7e72e4b52" # Put your key here
    monitor = HormuzMonitor(API_KEY)
    try:
        asyncio.run(monitor.run_monitor())
    except KeyboardInterrupt:
        print("\nStopping...")