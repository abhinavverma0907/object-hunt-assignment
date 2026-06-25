#!/usr/bin/env python3
"""
object_hunter.py
The Great Object Hunt — Session 4 Assignment (Sensors & Perception)
Pipeline: SEARCH -> DETECT -> TRACK -> APPROACH -> COMPLETE
"""

import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class ObjectHunter(Node):

    SAFE_STOP_DISTANCE = 0.6
    SEARCH_ANGULAR_SPEED = 0.4
    APPROACH_LINEAR_SPEED = 0.18
    CENTERING_TOLERANCE_PX = 25
    TURN_GAIN = 0.0025
    MISSING_FRAMES_BEFORE_SEARCH = 10
    DEPTH_PATCH_RADIUS = 4
    CONF_THRESHOLD = 0.35

    def __init__(self):
        super().__init__('object_hunter')

        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO("yolov8s.pt")
        self.get_logger().info("YOLO model loaded.")

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image, 'camera/image', self.rgb_callback, 1
        )
        self.depth_sub = self.create_subscription(
            Image, 'camera/depth_image', self.depth_callback, 1
        )
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None

        self.target_class = None
        self.status = "IDLE"
        self.distance = None
        self.missing_frames = 0
        self.mission_active = False
        self.running = True

        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.prev_time = time.time()

    def _spin_loop(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.lock:
            self.latest_rgb = frame

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self.lock:
            self.latest_depth = depth

    def _input_loop(self):
        while self.running:
            if not self.mission_active:
                try:
                    target = input("\nEnter target object (e.g. bottle, person, chair): ").strip().lower()
                except EOFError:
                    break
                if target == "":
                    continue
                if target in ("quit", "exit"):
                    self.running = False
                    break

                self.target_class = target
                self.missing_frames = 0
                self.mission_active = True
                self.status = "SEARCHING"
                print(f"\nSearching for: {self.target_class}")
            else:
                time.sleep(0.2)

    def estimate_distance(self, depth_frame, cx, cy):
        h, w = depth_frame.shape[:2]
        r = self.DEPTH_PATCH_RADIUS

        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        patch = depth_frame[y1:y2, x1:x2].astype(np.float32)

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def run_yolo_and_act(self, rgb_frame, depth_frame):
        twist = Twist()
        h, w = rgb_frame.shape[:2]
        frame_center_x = w // 2

        results = self.model(rgb_frame, conf=self.CONF_THRESHOLD, imgsz=640, verbose=False)

        target_boxes = []
        all_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[class_id]
                all_detections.append(f"{class_name} ({conf:.2f})")

                color = self._class_color(class_id)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(rgb_frame, f"{class_name} {conf:.2f}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if self.mission_active and class_name == self.target_class:
                    target_boxes.append((x1, y1, x2, y2, conf))

        if self.mission_active:
            if target_boxes:
                x1, y1, x2, y2, conf = max(target_boxes, key=lambda b: b[4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(rgb_frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                self.missing_frames = 0
                dist = self.estimate_distance(depth_frame, cx, cy) if depth_frame is not None else None

                if dist is not None:
                    self.distance = dist
                    print(f"Target Found")
                    print(f"Distance to {self.target_class}: {dist:.2f} m")

                    if dist <= self.SAFE_STOP_DISTANCE:
                        self.status = "STOPPED"
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        print("\nMission Completed")
                        print("Target Reached Successfully\n")

                        self._end_mission()

                    else:
                        self.status = "APPROACH"
                        pixel_error = frame_center_x - cx

                        if abs(pixel_error) > self.CENTERING_TOLERANCE_PX:
                            twist.angular.z = float(np.clip(self.TURN_GAIN * pixel_error, -0.6, 0.6))
                            twist.linear.x = self.APPROACH_LINEAR_SPEED * 0.4
                        else:
                            twist.angular.z = 0.0
                            twist.linear.x = min(self.APPROACH_LINEAR_SPEED, 0.12 * dist + 0.05)
                else:
                    self.status = "TRACKING"
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

            else:
                self.missing_frames += 1
                if self.missing_frames >= self.MISSING_FRAMES_BEFORE_SEARCH:
                    self.status = "SEARCHING"
                    twist.linear.x = 0.0
                    twist.angular.z = self.SEARCH_ANGULAR_SPEED
                    print("Searching...")
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

        if self.mission_active:
            self.cmd_pub.publish(twist)

        return rgb_frame, all_detections

    def _end_mission(self):
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.mission_active = False
        self.target_class = None
        self.distance = None
        self.missing_frames = 0
        self.status = "IDLE"

    def _draw_dashboard(self, frame, detections):
        dash_w = 380
        dash = np.zeros((frame.shape[0], dash_w, 3), dtype=np.uint8)

        cv2.putText(dash, "MISSION CONTROL", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        target_txt = self.target_class if self.target_class else "-- none --"
        rows = [
            (f"TARGET   : {target_txt}", (255, 255, 255)),
            (f"STATUS   : {self.status}", (0, 255, 0)),
            (f"DISTANCE : {f'{self.distance:.2f} m' if self.distance else '--'}", (0, 255, 0)),
            (f"MODE     : {self.status}", (255, 200, 0)),
        ]
        y = 80
        for text, color in rows:
            cv2.putText(dash, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 40

        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        cv2.putText(dash, f"FPS: {fps:.1f}", (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        y += 60
        cv2.putText(dash, f"All detections ({len(detections)}):", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1)
        y += 25
        for d in detections[:12]:
            cv2.putText(dash, d, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y += 22

        return np.hstack((frame, dash))

    @staticmethod
    def _class_color(class_id):
        np.random.seed(class_id)
        return tuple(int(c) for c in np.random.randint(100, 255, 3))

    def display_loop(self):
        cv2.namedWindow("Object Hunter", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Object Hunter", 1500, 850)

        while rclpy.ok() and self.running:
            with self.lock:
                rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()

            if rgb is not None:
                processed, detections = self.run_yolo_and_act(rgb, depth)
                combined = self._draw_dashboard(processed, detections)
                cv2.imshow("Object Hunter", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        zero = Twist()
        self.cmd_pub.publish(zero)
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectHunter()

    print("=" * 60)
    print("  THE GREAT OBJECT HUNT")
    print("  Type a target object below to begin the mission.")
    print("=" * 60)

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
