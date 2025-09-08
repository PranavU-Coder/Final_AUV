# ESSENTIAL IMPORTS
import cv2 as cv
import numpy as np
import time
import zmq
import pickle
import json

# USING THE OTHER PYTHON SCRIPTS FOR THIS MAIN IMPORTANT FILE
import triangulation as tri
import calibration

# COUPLING WITH THE YOLO MODEL
from ultralytics import YOLO



# AI SLOP IN ZMQ AND FSM-LIKE BEHAVIOUR CODE

# -----------------------
# Configuration
# -----------------------
MODEL_PATH = '/home/pranav/Projects/Final_AUV/torpedo_launcher/bounding_boxes/runs/detect/train/weights/best.pt'
SUB_ENDPOINT = "tcp://10.19.45.252:5555"  # RPi5 -> Laptop frames
PUB_ENDPOINT = "tcp://*:5566"             # Laptop -> RPi5 results

# Stereo and camera parameters
b = 5.15   # Distance between cameras (in cm)
f = 3.29   # Focal length (in mm)
alpha = 72.4  # Camera's FOV (degrees)

# Classes expected in the trained model
TARGET_CLASSES = ('shark', 'hole')

# -----------------------
# Helper functions
# -----------------------
def extract_detections(results, wanted=('shark', 'hole')):
    """
    Convert Ultralytics Results -> simple list of dicts per class:
    [{'box': (x1, y1, x2, y2), 'center': (cx, cy), 'conf': float, 'class': name}, ...]
    """
    out = {k: [] for k in wanted}
    if results and len(results) > 0:
        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                continue
            cls_np = boxes.cls.cpu().numpy().astype(int)
            xyxy_np = boxes.xyxy.cpu().numpy().astype(int)
            conf_np = boxes.conf.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(xyxy_np):
                name = r.names[int(cls_np[i])]
                if name in out:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    out[name].append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int(cx), int(cy)),
                        'conf': float(conf_np[i]),
                        'class': name
                    })
    return out

def pick_highest_conf(objs):
    """Return highest-confidence object dict from list or None."""
    if not objs:
        return None
    return max(objs, key=lambda d: d['conf'])

def nearest_to(ref_xy, candidates):
    """Return candidate dict whose center is nearest to ref_xy or None."""
    if not candidates:
        return None
    rx, ry = ref_xy
    best = None
    best_d2 = None
    for c in candidates:
        cx, cy = c['center']
        d2 = (cx - rx) ** 2 + (cy - ry) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best = c
    return best

def draw_annotations(img, shark, circle, color_shark=(0, 255, 255), color_circle=(0, 128, 255)):
    """Optional: draw boxes and centers for visualization."""
    if shark:
        x1, y1, x2, y2 = shark['box']
        cv.rectangle(img, (x1, y1), (x2, y2), color_shark, 2)
        cx, cy = shark['center']
        cv.circle(img, (cx, cy), 4, color_shark, -1)
        cv.putText(img, f"Shark {shark['conf']:.2f}", (x1, max(0, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color_shark, 2)
    if circle:
        x1, y1, x2, y2 = circle['box']
        cv.rectangle(img, (x1, y1), (x2, y2), color_circle, 2)
        cx, cy = circle['center']
        cv.circle(img, (cx, cy), 4, color_circle, -1)
        cv.putText(img, f"Circle {circle['conf']:.2f}", (x1, min(img.shape - 6, y2 + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color_circle, 2)

# -----------------------
# Main processing loop
# -----------------------
def main():
    # Load model
    model = YOLO(MODEL_PATH)

    # ZeroMQ setup
    context = zmq.Context()
    
    # Receive frames from RPi5
    frame_receiver = context.socket(zmq.SUB)
    frame_receiver.connect(SUB_ENDPOINT)
    frame_receiver.setsockopt_string(zmq.SUBSCRIBE, '')  # subscribe to all topics
    frame_receiver.setsockopt(zmq.RCVTIMEO, 5000)

    # Send results back to RPi5
    result_sender = context.socket(zmq.PUB)
    result_sender.bind(PUB_ENDPOINT)

    print("Laptop processor started, waiting for frames from RPi5...")
    time.sleep(3)

    frame_count = 0
    try:
        while True:
            try:
                print("Waiting for frame data...")
                frame_data = frame_receiver.recv()
                data = pickle.loads(frame_data)

                # Expecting keys: 'frame_right', 'frame_left', 'timestamp', 'frame_id'
                frame_right =cv.cvtColor(data['frame_right'],cv.COLOR_BGR2RGB)
                frame_left = cv.cvtColor(data['frame_left'],cv.COLOR_BGR2RGB)
                timestamp = data.get('timestamp', None)
                frame_id = data.get('frame_id', frame_count)

                frame_count += 1
                print(f"Received frame {frame_id} from RPi5 (total processed: {frame_count})")
                start = time.time()

                # CALIBRATION (undistort + rectify to make epipolar lines horizontal)
                try:
                    rect_left, rect_right = calibration.undistortRectify(frame_left, frame_right)
                    frame_left_proc, frame_right_proc = rect_left, rect_right
                    print("Calibration applied")
                except Exception as e:
                    print(f"Calibration failed, using raw frames: {e}")
                    frame_left_proc, frame_right_proc = frame_left, frame_right

                # YOLO PROCESSING
                print("Running YOLO inference...")
                results_right = model(frame_right_proc, verbose=False)
                results_left = model(frame_left_proc, verbose=False)

                det_right = extract_detections(results_right, wanted=TARGET_CLASSES)
                det_left = extract_detections(results_left, wanted=TARGET_CLASSES)

                # highest-confidence shark in each view
                shark_r = pick_highest_conf(det_right['shark'])
                shark_l = pick_highest_conf(det_left['shark'])

                # find the circle closest to the shark in each view
                circle_r = nearest_to(shark_r['center'], det_right['hole']) if shark_r else None
                circle_l = nearest_to(shark_l['center'], det_left['hole']) if shark_l else None

                print(f"Detections - Right(shark/hole): {len(det_right['shark'])}/{len(det_right['hole'])}, "
                      f"Left(shark/hole): {len(det_left['shark'])}/{len(det_left['hole'])}")

                # DEPTH CALCULATION: only for the circle nearest to the shark
                depth = None
                tracking_status = "LOST"
                if shark_r and shark_l and circle_r and circle_l:
                    try:
                        center_right = circle_r['center']
                        center_left = circle_l['center']
                        print(center_right,center_left)
                        depth = tri.find_depth(center_right, center_left,
                                               frame_right_proc, frame_left_proc,
                                               b, f, alpha)
                        tracking_status = "ACTIVE"
                        print(f"Calculated depth (nearest circle to shark): {round(depth, 1)} cm")
                    except Exception as e:
                        print(f"Depth calculation failed: {e}")
                        tracking_status = "LOST"
                else:
                    print("Missing shark or circle in one/both views; skipping depth for this frame.")

                # FPS
                end = time.time()
                processing_time = end - start
                fps = 1.0 / processing_time if processing_time > 0 else 0.0

                # Optional: annotate frames (comment out if headless)
                # draw_annotations(frame_right_proc, shark_r, circle_r)
                # draw_annotations(frame_left_proc, shark_l, circle_l)
                # cv.imshow("Right", frame_right_proc)
                # cv.imshow("Left", frame_left_proc)
                # if cv.waitKey(1) & 0xFF == 27:
                #     break

                # Build a compact result payload
                def to_simple_list(dlist):
                    return [{'box': d['box'], 'center': d['center'], 'class': d['class'], 'conf': d['conf']} for d in dlist]

                result_data = {
                    'timestamp': timestamp,
                    'frame_id': frame_id,
                    'tracking_status': tracking_status,
                    'depth': round(depth, 1) if depth is not None else None,
                    'fps': int(fps),
                    'shark_right': (shark_r and {'center': shark_r['center'], 'conf': shark_r['conf']}) or None,
                    'shark_left':  (shark_l and {'center': shark_l['center'],  'conf': shark_l['conf']}) or None,
                    'circle_right': (circle_r and {'center': circle_r['center'], 'conf': circle_r['conf']}) or None,
                    'circle_left':  (circle_l and {'center': circle_l['center'],  'conf': circle_l['conf']}) or None,
                    'all_right': { 'shark': to_simple_list(det_right['shark']), 'hole': to_simple_list(det_right['hole']) },
                    'all_left':  { 'shark': to_simple_list(det_left['shark']),  'hole': to_simple_list(det_left['hole']) },
                    'processing_time_ms': round(processing_time * 1000, 2)
                }

                result_sender.send(pickle.dumps(result_data))
                print(f"Sent results back - Status: {tracking_status}, Depth: {result_data['depth']} cm, FPS: {result_data['fps']}")
                print("-" * 60)

            except zmq.Again:
                print("Timeout waiting for frames from RPi5...")
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)
                continue

    except KeyboardInterrupt:
        print("\nStopping laptop processor...")
    finally:
        try:
            frame_receiver.close()
        except Exception:
            pass
        try:
            result_sender.close()
        except Exception:
            pass
        try:
            context.term()
        except Exception:
            pass
        print("Cleanup complete")

if __name__ == "__main__":
    main()

