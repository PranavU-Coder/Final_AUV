import cv2 as cv
import numpy as np
import time
import zmq
import pickle
import json
from picamera2 import Picamera2
import os
import calibration


# AI SLOP TO REWRITE & REWORK IN FUTURE

# ZMQ setup
context = zmq.Context()

# Publisher: frames to laptop
frame_sender = context.socket(zmq.PUB)
frame_sender.bind("tcp://*:5555")  # RPi5 binds, laptop SUB connects
# mitigate slow joiner: give subscribers time to subscribe before first send
time.sleep(0.5)

# Subscriber: results from laptop
result_receiver = context.socket(zmq.SUB)
# Replace with your laptop IP that runs the processing script
result_receiver.connect("tcp://10.19.45.235:5566")
result_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
result_receiver.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout to keep loop responsive
# allow publisher time to bind and SUB to propagate subscriptions
time.sleep(0.5)

# Cameras
camL = Picamera2(0)
camR = Picamera2(1)

# Tip: set 'BGR888' to skip cvtColor if desired
config_L = camL.create_preview_configuration({"format": "RGB888"})
config_R = camR.create_preview_configuration({"format": "RGB888"})
camL.configure(config_L)
camR.configure(config_R)
camL.start()
camR.start()

# Latest results default aligned to new payload structure
latest_results = {
    "tracking_status": "WAITING",
    "depth": None,
    "fps": 0,
    "shark_right": None,
    "shark_left": None,
    "circle_right": None,
    "circle_left": None,
    "all_right": {"shark": [], "hole": []},
    "all_left": {"shark": [], "hole": []},
    "processing_time_ms": None
}

def _draw_all_dets(img, all_dets, color_map=None, thickness=2):
    """
    Draw all detections from all_dets = {'shark': [...], 'hole': [...]}
    Each item has keys: 'box', 'center', 'class', 'conf'
    """
    if color_map is None:
        color_map = {"shark": (0, 255, 255), "hole": (0, 200, 255)}
    for cls_name, dets in all_dets.items():
        for d in dets:
            (x1, y1, x2, y2) = d.get("box", (None, None, None, None))
            if x1 is None:
                continue
            cx, cy = d.get("center", (None, None))
            conf = d.get("conf", 0.0)
            color = color_map.get(cls_name, (0, 255, 0))
            cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            if cx is not None:
                cv.circle(img, (cx, cy), 4, color, -1)
            cv.putText(img, f"{cls_name} {conf:.2f}", (x1, max(0, y1 - 6)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def _highlight_choice(img, choice_center, all_dets, color=(0, 0, 255), thickness=3):
    """
    Highlight the chosen circle/shark if present:
    - If a center is provided, draw a larger marker at that center
    - Try to find a matching detection by center to draw a thicker box
    """
    if not choice_center:
        return
    cx_sel, cy_sel = choice_center
    cv.circle(img, (cx_sel, cy_sel), 6, color, -1)
    # attempt to find exact center match among holes first, then sharks
    for cls_name in ("hole", "shark"):
        for d in all_dets.get(cls_name, []):
            cx, cy = d.get("center", (None, None))
            if cx == cx_sel and cy == cy_sel and d.get("box"):
                x1, y1, x2, y2 = d["box"]
                cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                return

def draw_results(frame_right, frame_left, results):
    # Draw all detections for context
    all_r = results.get("all_right", {"shark": [], "hole": []})
    all_l = results.get("all_left", {"shark": [], "hole": []})
    _draw_all_dets(frame_right, all_r)
    _draw_all_dets(frame_left, all_l)

    # Emphasize the selected nearest circle and the shark (if centers provided)
    circle_r = results.get("circle_right") or {}
    circle_l = results.get("circle_left") or {}
    shark_r = results.get("shark_right") or {}
    shark_l = results.get("shark_left") or {}

    _highlight_choice(frame_right, circle_r.get("center"), all_r, color=(0, 0, 255), thickness=3)
    _highlight_choice(frame_left,  circle_l.get("center"), all_l, color=(0, 0, 255), thickness=3)

    _highlight_choice(frame_right, shark_r.get("center"), all_r, color=(0, 255, 0), thickness=3)
    _highlight_choice(frame_left,  shark_l.get("center"), all_l, color=(0, 255, 0), thickness=3)

    # Status overlays
    status = results.get("tracking_status", "WAITING")
    depth = results.get("depth", None)
    fps = results.get("fps", 0)

    if status == "LOST":
        cv.putText(frame_right, "TRACKING LOST", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv.putText(frame_left,  "TRACKING LOST", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif depth is not None:
        cv.putText(frame_right, f"Distance: {depth} cm", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv.putText(frame_left,  f"Distance: {depth} cm", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    height_r = frame_right.shape[0]
    height_l = frame_left.shape[0]

    cv.putText(frame_right, f"FPS: {fps}", (20, height_r - 20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv.putText(frame_left,  f"FPS: {fps}", (20, height_l - 20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def main():
    global latest_results

    print("RPi5 capture started, sending frames to laptop...")
    time.sleep(1.0)  # connection warm-up for pub/sub

    try:
        for frame_count in range(10):
            print(f"Processing frame {frame_count + 1}/10")

            # Capture
            frame_right = camR.capture_array()
            frame_left = camL.capture_array()

            # Convert from RGB to BGR for OpenCV operations
            frame_right_bgr = cv.cvtColor(frame_right, cv.COLOR_RGB2BGR)
            frame_left_bgr = cv.cvtColor(frame_left, cv.COLOR_RGB2BGR)

            # Optional: pre-rectify with feature-based method if desired
            try:
                # frame_left_bgr, frame_right_bgr = calibration.orb_rect(frame_left_bgr, frame_right_bgr)
                pass
            except Exception as e:
                # If this is experimental, it's safer to proceed with raw frames
                print(f"orb_rect failed or not configured, using raw frames: {e}")

            # Prepare and send frame packet
            frame_data = {
                "frame_right": frame_right_bgr,
                "frame_left": frame_left_bgr,
                "timestamp": time.time(),
                "frame_id": frame_count
            }

            try:
                frame_bytes = pickle.dumps(frame_data)
                frame_sender.send(frame_bytes)
                print(f"Sent frame {frame_count} to laptop")
            except Exception as e:
                print(f"Error sending frames: {e}")

            # Receive results (non-blocking loop with short timeout)
            max_wait_attempts = 5
            received_result = False
            for attempt in range(max_wait_attempts):
                try:
                    result_data = result_receiver.recv()
                    latest_results = pickle.loads(result_data)
                    print(f"Received results for frame {frame_count}")
                    print(f"   Status: {latest_results.get('tracking_status')}")
                    if latest_results.get("depth") is not None:
                        print(f"   Depth: {latest_results['depth']} cm")
                    received_result = True
                    break
                except zmq.Again:
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"Error receiving results: {e}")
                    break

            if not received_result:
                print(f"No new results received for frame {frame_count}")

            # Draw overlays on the same frames that were sent/saved
            draw_results(frame_right_bgr, frame_left_bgr, latest_results)

            # Save frames with annotation overlay
            cv.imwrite(f"right_{frame_count:02d}.jpg", frame_right_bgr)
            cv.imwrite(f"left_{frame_count:02d}.jpg", frame_left_bgr)
            print(f"Saved frame {frame_count} with detection results")

            # Small delay
            time.sleep(0.2)

    finally:
        print("Cleaning up...")
        try:
            camL.stop()
            camR.stop()
            print("Cameras stopped")
        except Exception:
            pass

        try:
            frame_sender.close()
            result_receiver.close()
            context.term()
            print("ZMQ connections closed")
        except Exception:
            pass

if _name_ == "_main_":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping RPi5 capture...")
    finally:
        try:
            camL.stop()
            camR.stop()
        except Exception:
            pass
