import cv2
import pandas as pd


EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot_index"),
]


def annotate_video(video_path, df: pd.DataFrame, output_path: str, left_ic=None, right_ic=None, left_to=None, right_to=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    lookup = df.set_index("frame")

    left_ic = set(left_ic if left_ic is not None else [])
    right_ic = set(right_ic if right_ic is not None else [])
    left_to = set(left_to if left_to is not None else [])
    right_to = set(right_to if right_to is not None else [])

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1


        if idx in left_ic:
            cv2.putText(
                frame,
                f"LEFT IC  frame={idx}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if idx in right_ic:
            cv2.putText(
                frame,
                f"RIGHT IC  frame={idx}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if idx in left_to:
            cv2.putText(
                frame,
                f"LEFT TO frame={idx}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if idx in right_to:
            cv2.putText(
                frame,
                f"RIGHT TO frame={idx}",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if idx in lookup.index:
            row = lookup.loc[idx]

            # Рисуем скелет
            for a, b in EDGES:
                ax, ay = row[f"{a}_x"] * w, row[f"{a}_y"] * h
                bx, by = row[f"{b}_x"] * w, row[f"{b}_y"] * h

                if all(map(lambda z: z == z, [ax, ay, bx, by])):
                    cv2.line(
                        frame,
                        (int(ax), int(ay)),
                        (int(bx), int(by)),
                        (255, 255, 255),
                        2,
                    )

            # Рисуем суставы
            for name in set(sum(([a, b] for a, b in EDGES), [])):
                x, y = row[f"{name}_x"] * w, row[f"{name}_y"] * h

                if x == x and y == y:
                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        4,
                        (255, 255, 255),
                        -1,
                    )

            # ==========================================
            # УГОЛ ЛЕВОГО КОЛЕНА
            # ==========================================
            left_knee_angle = row["left_knee_angle"]

            if pd.notna(left_knee_angle):
                knee_x = int(row["left_knee_x"] * w)
                knee_y = int(row["left_knee_y"] * h)

                cv2.putText(
                    frame,
                    f"{left_knee_angle:.1f} deg",
                    (knee_x + 10, knee_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # ==========================================
            # УГОЛ ПРАВОГО КОЛЕНА
            # ==========================================
            right_knee_angle = row["right_knee_angle"]

            if pd.notna(right_knee_angle):
                knee_x = int(row["right_knee_x"] * w)
                knee_y = int(row["right_knee_y"] * h)

                cv2.putText(
                    frame,
                    f"{right_knee_angle:.1f} deg",
                    (knee_x + 10, knee_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(frame)

    cap.release()
    writer.release()
