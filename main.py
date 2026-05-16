import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from time import perf_counter
from mediapipe.python.solutions import hands as mp_hands

# --- INITIALIZATION ---
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)

# --- PHYSICS SETTINGS ---
GRAVITY = 1.2
BOUNCE = -0.3
FLOOR_Y = 380
box_size = 70
PINCH_THRESHOLD = 0.035
GRAB_MARGIN = 18
SHATTER_SPEED = 28
FRAGMENT_LIFETIME = 1.0
THROW_SCALE = 0.04
MAX_THROW_SPEED = 35
PALM_CONNECTIONS = {
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (5, 9), (9, 13), (13, 17)
}
FINGERTIPS = {4, 8, 12, 16, 20}
materials = [
    {"name": "glass", "color": (210, 235, 255), "gravity": 1.0, "bounce": -0.2},
    {"name": "rubber", "color": (80, 230, 120), "gravity": 1.2, "bounce": -0.85},
    {"name": "metal", "color": (180, 185, 195), "gravity": 2.0, "bounce": -0.08}
]

# --- BOX SETUP ---
boxes = []
num_boxes = 3
spawn_positions_ready = False
for i in range(num_boxes):
    start_pos = [80 + (i * 160), 50]
    boxes.append({
        "id": i + 1,
        "material": materials[i],
        "start_pos": start_pos.copy(),
        "pos": start_pos.copy(),
        "vel": [0, 0],
        "is_grabbing": False,
        "grabbed_by": None,
        "grab_started_at": None,
        "grab_timestamp": None,
        "hand_prev_pos": None,
        "hand_prev_time": None,
        "release_velocity": [0, 0],
        "dropped_by_user": False,
        "active": True
    })

# --- GRAB LOGGING ---
grab_events = []
fragments = []


def set_spawn_positions(screen_width):
    spacing = screen_width // (num_boxes + 1)
    for i, box in enumerate(boxes):
        start_x = spacing * (i + 1) - box_size // 2
        start_x = max(0, min(screen_width - box_size, start_x))
        box["start_pos"] = [start_x, 50]
        box["pos"] = box["start_pos"].copy()


def finish_grab(box, apply_throw=True):
    if box["grab_started_at"] is None:
        return

    grab_events.append({
        "timestamp": box["grab_timestamp"],
        "box": f"Box {box['id']}",
        "held_seconds": round(perf_counter() - box["grab_started_at"], 2)
    })
    if apply_throw:
        box["vel"] = box["release_velocity"].copy()
        box["dropped_by_user"] = True

    box["grab_started_at"] = None
    box["grab_timestamp"] = None
    box["hand_prev_pos"] = None
    box["hand_prev_time"] = None
    box["release_velocity"] = [0, 0]
    box["is_grabbing"] = False
    box["grabbed_by"] = None


def show_grab_chart():
    if not grab_events:
        print("No grab events were recorded.")
        return

    df = pd.DataFrame(grab_events)
    print("\nGrab Event Log:")
    print(df.to_string(index=False))

    grab_counts = df["box"].value_counts().reindex(
        [f"Box {i + 1}" for i in range(num_boxes)],
        fill_value=0
    )

    grab_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Grab Counts by Box")
    plt.xlabel("Box")
    plt.ylabel("Number of Grabs")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def reset_boxes():
    fragments.clear()
    for box in boxes:
        finish_grab(box, apply_throw=False)
        box["pos"] = box["start_pos"].copy()
        box["vel"] = [0, 0]
        box["is_grabbing"] = False
        box["grabbed_by"] = None
        box["grab_started_at"] = None
        box["grab_timestamp"] = None
        box["hand_prev_pos"] = None
        box["hand_prev_time"] = None
        box["release_velocity"] = [0, 0]
        box["dropped_by_user"] = False
        box["active"] = True


def shatter_box(box):
    finish_grab(box, apply_throw=False)
    box["active"] = False
    box["is_grabbing"] = False

    now = perf_counter()
    center_x = box["pos"][0] + box_size // 2
    center_y = box["pos"][1] + box_size // 2
    shard_size = box_size // 4

    for row in range(3):
        for col in range(3):
            fragments.append({
                "pos": [
                    center_x - shard_size + col * shard_size,
                    center_y - shard_size + row * shard_size
                ],
                "vel": [(col - 1) * 3, -8 + row * 2],
                "size": shard_size,
                "created_at": now,
                "color": box["material"]["color"]
            })


while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    display = np.zeros((h, w, 3), dtype=np.uint8)

    if not spawn_positions_ready:
        set_spawn_positions(w)
        spawn_positions_ready = True

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    active_fingers = set()
    detected_hands = set()

    # 1. HAND PROCESSING & INTERACTION
    if results.multi_hand_landmarks:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            detected_hands.add(hand_label)
            lms = hand_lms.landmark
            ix, iy = int(lms[8].x * w), int(lms[8].y * h)

            # Pinch Check
            dist = math.sqrt((lms[4].x - lms[8].x) ** 2 + (lms[4].y - lms[8].y) ** 2)

            if dist < PINCH_THRESHOLD:
                active_fingers.add(hand_label)

                held_box = None
                for box in boxes:
                    if box["active"] and box["grabbed_by"] == hand_label:
                        held_box = box
                        break

                if held_box is None:
                    for box in reversed(boxes):
                        if box["active"] and box["grabbed_by"] is None and \
                                box["pos"][0] - GRAB_MARGIN < ix < box["pos"][0] + box_size + GRAB_MARGIN and \
                                box["pos"][1] - GRAB_MARGIN < iy < box["pos"][1] + box_size + GRAB_MARGIN:
                            held_box = box
                            break

                if held_box is not None:
                    now = perf_counter()
                    if held_box["grab_started_at"] is None:
                        held_box["grab_started_at"] = now
                        held_box["grab_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        held_box["hand_prev_pos"] = [ix, iy]
                        held_box["hand_prev_time"] = now
                        held_box["release_velocity"] = [0, 0]
                        held_box["dropped_by_user"] = False

                    if held_box["hand_prev_pos"] is not None and held_box["hand_prev_time"] is not None:
                        dt = max(now - held_box["hand_prev_time"], 0.001)
                        vx = (ix - held_box["hand_prev_pos"][0]) / dt * THROW_SCALE
                        vy = (iy - held_box["hand_prev_pos"][1]) / dt * THROW_SCALE
                        held_box["release_velocity"] = [
                            max(-MAX_THROW_SPEED, min(MAX_THROW_SPEED, vx)),
                            max(-MAX_THROW_SPEED, min(MAX_THROW_SPEED, vy))
                        ]
                        held_box["hand_prev_pos"] = [ix, iy]
                        held_box["hand_prev_time"] = now

                    held_box["is_grabbing"] = True
                    held_box["grabbed_by"] = hand_label
                    held_box["vel"] = [0, 0]
                    held_box["pos"][0] = ix - box_size // 2
                    held_box["pos"][1] = iy - box_size // 2

                    # Z-Layering: Move grabbed box to end of list
                    if boxes[-1] is not held_box:
                        boxes.remove(held_box)
                        boxes.append(held_box)
            else:
                for box in boxes:
                    if box["active"] and box["grabbed_by"] == hand_label:
                        finish_grab(box)

            # Draw Hand Skeleton
            hand_color = (0, 255, 255) if hand_label in active_fingers else (255, 255, 255)
            for connection in mp_hands.HAND_CONNECTIONS:
                start = int(connection[0])
                end = int(connection[1])
                p1 = (int(lms[start].x * w), int(lms[start].y * h))
                p2 = (int(lms[end].x * w), int(lms[end].y * h))
                thickness = 4 if (start, end) in PALM_CONNECTIONS or (end, start) in PALM_CONNECTIONS else 2
                cv2.line(display, p1, p2, hand_color, thickness)

            for landmark_id, landmark in enumerate(lms):
                point = (int(landmark.x * w), int(landmark.y * h))
                radius = 6 if landmark_id in FINGERTIPS else 3
                cv2.circle(display, point, radius, hand_color, cv2.FILLED)

    for box in boxes:
        if box["active"] and box["grabbed_by"] is not None and box["grabbed_by"] not in detected_hands:
            finish_grab(box)

    # 2. PHYSICS & SOLID COLLISION RESOLUTION
    cv2.line(display, (0, FLOOR_Y + box_size), (w, FLOOR_Y + box_size), (70, 70, 70), 1)

    for i, box in enumerate(boxes):
        if not box["active"]:
            continue

        if not box["is_grabbing"]:
            # Apply Gravity
            box["vel"][1] += box["material"]["gravity"]
            box["pos"][0] += int(box["vel"][0])
            box["pos"][1] += int(box["vel"][1])

            if box["pos"][0] < 0:
                box["pos"][0] = 0
                box["vel"][0] *= -0.4
            elif box["pos"][0] > w - box_size:
                box["pos"][0] = w - box_size
                box["vel"][0] *= -0.4

            # BOX vs BOX SOLID COLLISION
            for j, other in enumerate(boxes):
                if i == j: continue
                if not other["active"]: continue

                # AABB Collision Detection
                if (box["pos"][0] < other["pos"][0] + box_size and
                        box["pos"][0] + box_size > other["pos"][0] and
                        box["pos"][1] < other["pos"][1] + box_size and
                        box["pos"][1] + box_size > other["pos"][1]):

                    # Resolve Vertical (Stacking)
                    if box["pos"][1] < other["pos"][1]:  # Coming from above
                        box["pos"][1] = other["pos"][1] - box_size
                        box["vel"][1] *= box["material"]["bounce"]
                        box["vel"][0] *= 0.96
                        if abs(box["vel"][1]) < 2: box["vel"][1] = 0
                    else:  # Coming from below or side-shuffle
                        box["pos"][1] = other["pos"][1] + box_size
                        box["vel"][1] = 1  # Push down slightly

            # Floor Collision
            if box["pos"][1] > FLOOR_Y:
                impact_speed = box["vel"][1]
                if box["material"]["name"] == "glass" and box["dropped_by_user"] and impact_speed > SHATTER_SPEED:
                    shatter_box(box)
                    continue

                box["pos"][1] = FLOOR_Y
                box["vel"][1] *= box["material"]["bounce"]
                box["vel"][0] *= 0.98
                if abs(box["vel"][1]) < 1.5: box["vel"][1] = 0
                if abs(box["vel"][0]) < 0.2: box["vel"][0] = 0

    now = perf_counter()
    for fragment in fragments[:]:
        age = now - fragment["created_at"]
        if age > FRAGMENT_LIFETIME:
            fragments.remove(fragment)
            continue

        fragment["vel"][1] += GRAVITY
        fragment["pos"][0] += int(fragment["vel"][0])
        fragment["pos"][1] += int(fragment["vel"][1])

        fade = max(0.0, 1.0 - age / FRAGMENT_LIFETIME)
        color = tuple(int(channel * fade) for channel in fragment["color"])
        x, y = fragment["pos"]
        size = fragment["size"]
        cv2.rectangle(display, (x, y), (x + size, y + size), color, cv2.FILLED)

    # 3. RENDER ALL BOXES
    for box in boxes:
        if not box["active"]:
            continue

        x, y = box["pos"]
        color = box["material"]["color"]
        border_color = (255, 255, 255) if box["is_grabbing"] else (180, 180, 180)

        cv2.rectangle(display, (x, y), (x + box_size, y + box_size), color, cv2.FILLED)
        cv2.rectangle(display, (x, y), (x + box_size, y + box_size), border_color, 2)

        label = str(box["id"])
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_x = x + (box_size - text_size[0]) // 2
        text_y = y + (box_size + text_size[1]) // 2
        cv2.putText(display, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

        material_label = box["material"]["name"]
        material_size, _ = cv2.getTextSize(material_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        material_x = x + (box_size - material_size[0]) // 2
        material_y = y + box_size - 10
        cv2.putText(display, material_label, (material_x, material_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)

    # UI
    pinch_status = "Pinching: " + ", ".join(sorted(active_fingers)) if active_fingers else "Pinching: None"
    cv2.putText(display, "Physics Sandbox", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
    cv2.putText(display, pinch_status, (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Hand Analysis Core", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        reset_boxes()

for box in boxes:
    finish_grab(box)

cap.release()
cv2.destroyAllWindows()
show_grab_chart()
