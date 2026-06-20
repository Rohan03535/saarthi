"""
Generate a synthetic CCTV-style traffic video for SAARTHI Edge-AI demo.
The video simulates a top-down camera at a Bengaluru intersection.
Traffic starts normal, then gradually builds to complete gridlock.
"""
import cv2
import numpy as np
import random
import os

# ── Video Settings ────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 600
FPS = 20
DURATION_SEC = 28
TOTAL_FRAMES = FPS * DURATION_SEC
OUT_PATH = os.path.join(os.path.dirname(__file__), "traffic_cam.mp4")

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
GROUND      = (30, 38, 30)
ROAD_H      = (55, 55, 55)
ROAD_V      = (55, 55, 55)
LANE_MARK   = (75, 75, 75)    # Dim enough to stay below detection threshold (90)
SIDEWALK    = (55, 60, 50)
CAR_COLORS  = [
    (35, 35, 200),   # Red
    (200, 45, 45),   # Blue
    (0, 180, 220),   # Yellow
    (45, 190, 45),   # Green
    (200, 110, 40),  # Teal/Cyan
    (60, 60, 180),   # Dark Red
    (180, 60, 120),  # Purple
    (40, 140, 200),  # Orange
    (100, 200, 200), # Light yellow
    (200, 100, 180), # Pink
]

# ── Road Layout ───────────────────────────────────────────────────────────────
# Vertical road (main artery) — lanes at these x-positions
V_ROAD_LEFT  = 300
V_ROAD_RIGHT = 500
V_LANES_DOWN = [330, 370]   # traffic going down (left side of road)
V_LANES_UP   = [430, 470]   # traffic going up (right side)

# Horizontal road — lanes at these y-positions
H_ROAD_TOP    = 220
H_ROAD_BOTTOM = 380
H_LANES_RIGHT = [250, 290]  # traffic going right
H_LANES_LEFT  = [330, 360]  # traffic going left


class Vehicle:
    def __init__(self, lane_pos, start_pos, direction, axis):
        self.axis = axis          # 'v' vertical or 'h' horizontal
        self.direction = direction  # +1 or -1
        self.color = random.choice(CAR_COLORS)

        if axis == 'v':
            self.x = lane_pos + random.randint(-4, 4)
            self.y = start_pos
            self.w = random.randint(22, 28)
            self.h = random.randint(38, 52)
        else:
            self.x = start_pos
            self.y = lane_pos + random.randint(-4, 4)
            self.w = random.randint(38, 52)
            self.h = random.randint(22, 28)

        self.base_speed = random.uniform(2.5, 5.0)
        self.speed = self.base_speed
        self.alive = True

    def update(self, congestion):
        """Move the vehicle; congestion is 0.0 (free) to 1.0 (stopped)."""
        effective = self.base_speed * max(0.0, 1.0 - congestion)
        # Add micro-jitter so stopped traffic doesn't look frozen
        if congestion > 0.85:
            effective += random.uniform(-0.15, 0.15)

        if self.axis == 'v':
            self.y += effective * self.direction
            if self.y < -80 or self.y > HEIGHT + 80:
                self.alive = False
        else:
            self.x += effective * self.direction
            if self.x < -80 or self.x > WIDTH + 80:
                self.alive = False

    def draw(self, frame):
        x1 = int(self.x - self.w // 2)
        y1 = int(self.y - self.h // 2)
        x2 = int(self.x + self.w // 2)
        y2 = int(self.y + self.h // 2)
        # Car body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        # Outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), 1)
        # Windshield highlight
        if self.axis == 'v':
            wy1 = y1 + 3 if self.direction == 1 else y2 - 8
            cv2.rectangle(frame, (x1 + 3, wy1), (x2 - 3, wy1 + 5),
                          (min(self.color[0]+60, 255), min(self.color[1]+60, 255), min(self.color[2]+60, 255)), -1)
        else:
            wx1 = x1 + 3 if self.direction == 1 else x2 - 8
            cv2.rectangle(frame, (wx1, y1 + 3), (wx1 + 5, y2 - 3),
                          (min(self.color[0]+60, 255), min(self.color[1]+60, 255), min(self.color[2]+60, 255)), -1)


def draw_road(frame, frame_idx):
    """Draw the intersection, lane markings, and sidewalks."""
    frame[:] = GROUND

    # Sidewalks
    cv2.rectangle(frame, (V_ROAD_LEFT - 10, 0), (V_ROAD_LEFT, HEIGHT), SIDEWALK, -1)
    cv2.rectangle(frame, (V_ROAD_RIGHT, 0), (V_ROAD_RIGHT + 10, HEIGHT), SIDEWALK, -1)
    cv2.rectangle(frame, (0, H_ROAD_TOP - 10), (WIDTH, H_ROAD_TOP), SIDEWALK, -1)
    cv2.rectangle(frame, (0, H_ROAD_BOTTOM), (WIDTH, H_ROAD_BOTTOM + 10), SIDEWALK, -1)

    # Roads
    cv2.rectangle(frame, (V_ROAD_LEFT, 0), (V_ROAD_RIGHT, HEIGHT), ROAD_V, -1)
    cv2.rectangle(frame, (0, H_ROAD_TOP), (WIDTH, H_ROAD_BOTTOM), ROAD_H, -1)

    # Intersection fill
    cv2.rectangle(frame, (V_ROAD_LEFT, H_ROAD_TOP), (V_ROAD_RIGHT, H_ROAD_BOTTOM), (50, 50, 50), -1)

    # Dashed lane markings — vertical road
    dash_len = 20
    gap_len = 15
    center_x = (V_ROAD_LEFT + V_ROAD_RIGHT) // 2
    offset = (frame_idx * 2) % (dash_len + gap_len)
    for y in range(-dash_len, HEIGHT + dash_len, dash_len + gap_len):
        y_start = y - offset
        # Center line (double yellow)
        cv2.line(frame, (center_x - 2, y_start), (center_x - 2, y_start + dash_len), (0, 180, 220), 2)
        cv2.line(frame, (center_x + 2, y_start), (center_x + 2, y_start + dash_len), (0, 180, 220), 2)
        # Lane dashes
        for lx in V_LANES_DOWN[1:] + V_LANES_UP[:-1]:
            if abs(lx - center_x) > 15:
                cv2.line(frame, (lx + 15, y_start), (lx + 15, y_start + dash_len), LANE_MARK, 1)

    # Dashed lane markings — horizontal road
    center_y = (H_ROAD_TOP + H_ROAD_BOTTOM) // 2
    for x in range(-dash_len, WIDTH + dash_len, dash_len + gap_len):
        x_start = x - offset
        cv2.line(frame, (x_start, center_y - 2), (x_start + dash_len, center_y - 2), (0, 180, 220), 2)
        cv2.line(frame, (x_start, center_y + 2), (x_start + dash_len, center_y + 2), (0, 180, 220), 2)

    # Crosswalk stripes at intersection edges
    for cx in range(V_ROAD_LEFT + 5, V_ROAD_RIGHT - 5, 12):
        cv2.rectangle(frame, (cx, H_ROAD_TOP - 8), (cx + 6, H_ROAD_TOP), (75, 75, 75), -1)
        cv2.rectangle(frame, (cx, H_ROAD_BOTTOM), (cx + 6, H_ROAD_BOTTOM + 8), (75, 75, 75), -1)
    for cy in range(H_ROAD_TOP + 5, H_ROAD_BOTTOM - 5, 12):
        cv2.rectangle(frame, (V_ROAD_LEFT - 8, cy), (V_ROAD_LEFT, cy + 6), (75, 75, 75), -1)
        cv2.rectangle(frame, (V_ROAD_RIGHT, cy), (V_ROAD_RIGHT + 8, cy + 6), (75, 75, 75), -1)


def add_overlay(frame, frame_idx, density):
    """Add CCTV-style overlay text."""
    h, w = frame.shape[:2]

    # Timestamp
    seconds = frame_idx / FPS
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    ts_text = f"2024-03-07  16:{37+mins:02d}:{secs:02d} IST"
    cv2.putText(frame, ts_text, (w - 290, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    # Camera ID
    cv2.putText(frame, "CAM-07  HOSUR ROAD JN", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    # REC indicator (blinking)
    if (frame_idx // 10) % 2 == 0:
        cv2.circle(frame, (w - 30, 22), 6, (0, 0, 200), -1)
        cv2.putText(frame, "REC", (w - 70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)

    # Density bar at bottom
    bar_x, bar_y, bar_w, bar_h = 15, h - 35, 200, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * density)
    bar_color = (0, 200, 0) if density < 0.6 else ((0, 180, 220) if density < 0.85 else (0, 0, 220))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
    cv2.putText(frame, f"DENSITY {density*100:.0f}%", (bar_x + bar_w + 10, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1, cv2.LINE_AA)

    # Add noise grain
    noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
    cv2.add(frame, noise, frame)


def main():
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    vehicles = []
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    spawn_interval_start = 8   # frames between spawns at start
    spawn_interval_end = 2     # frames between spawns at peak

    for fi in range(TOTAL_FRAMES):
        t = fi / TOTAL_FRAMES  # 0.0 → 1.0

        # ── Congestion curve ──────────────────────────────────────────────
        if t < 0.35:
            congestion = t * 0.3           # 0 → 0.10
        elif t < 0.65:
            congestion = 0.10 + (t - 0.35) * 2.3  # 0.10 → 0.79
        else:
            congestion = 0.79 + (t - 0.65) * 0.6   # 0.79 → 1.0

        congestion = min(congestion, 1.0)

        # ── Spawn vehicles ────────────────────────────────────────────────
        spawn_interval = max(2, int(spawn_interval_start - (spawn_interval_start - spawn_interval_end) * t))
        if fi % spawn_interval == 0:
            # Vertical lanes
            for lane in V_LANES_DOWN:
                if random.random() < 0.6:
                    vehicles.append(Vehicle(lane, -60, +1, 'v'))
            for lane in V_LANES_UP:
                if random.random() < 0.6:
                    vehicles.append(Vehicle(lane, HEIGHT + 60, -1, 'v'))
            # Horizontal lanes
            for lane in H_LANES_RIGHT:
                if random.random() < 0.4:
                    vehicles.append(Vehicle(lane, -60, +1, 'h'))
            for lane in H_LANES_LEFT:
                if random.random() < 0.4:
                    vehicles.append(Vehicle(lane, WIDTH + 60, -1, 'h'))

        # ── Update ────────────────────────────────────────────────────────
        for v in vehicles:
            v.update(congestion)
        vehicles = [v for v in vehicles if v.alive]

        # ── Draw ──────────────────────────────────────────────────────────
        draw_road(frame, fi)
        for v in vehicles:
            v.draw(frame)

        # Count visible vehicles for density display
        visible = sum(1 for v in vehicles
                      if 0 <= v.x <= WIDTH and 0 <= v.y <= HEIGHT)
        density = min(visible / 60.0, 1.0)  # 60 vehicles = 100% density

        add_overlay(frame, fi, density)

        # If gridlock, flash warning
        if congestion > 0.85 and (fi // 8) % 2 == 0:
            cv2.putText(frame, "!! GRIDLOCK DETECTED !!", (WIDTH // 2 - 160, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(frame)

        if fi % 100 == 0:
            print(f"  Frame {fi}/{TOTAL_FRAMES}  congestion={congestion:.2f}  vehicles={len(vehicles)}")

    writer.release()
    print(f"\nVideo saved: {OUT_PATH}  ({TOTAL_FRAMES} frames, {DURATION_SEC}s)")


if __name__ == "__main__":
    main()
