import cv2
import pandas as pd
import mediapipe as mp


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_video(self, path: str) -> pd.DataFrame:
        cap = cv2.VideoCapture(path)
        rows = []
        frame_idx = 0

        names = [
            "nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "left_heel", "right_heel",
            "left_foot_index", "right_foot_index",
        ]

        indices = {
            name: getattr(self.mp_pose.PoseLandmark, name.upper()).value
            for name in names
        }

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            row = {"frame": frame_idx}
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                for name, idx in indices.items():
                    p = lm[idx]
                    row[f"{name}_x"] = p.x
                    row[f"{name}_y"] = p.y
                    row[f"{name}_z"] = p.z
                    row[f"{name}_vis"] = p.visibility
            else:
                for name in names:
                    row[f"{name}_x"] = float("nan")
                    row[f"{name}_y"] = float("nan")
                    row[f"{name}_z"] = float("nan")
                    row[f"{name}_vis"] = 0.0

            rows.append(row)
            frame_idx += 1

        cap.release()
        return pd.DataFrame(rows)

    def close(self):
        self.pose.close()
