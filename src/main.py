import argparse
from pathlib import Path

from src.video.loader import get_video_info
from src.pose.detector import PoseDetector
from src.biomechanics.metrics import compute_metrics, smooth_landmarks
from src.biomechanics.gait_cycle import detect_initial_contacts, detect_toe_off
from src.visualization.plots import (
    save_plots,
    save_ic_plot,
    save_gait_events_plot,
)
from src.visualization.skeleton import annotate_video


def main():
    parser = argparse.ArgumentParser(description="Running biomechanics MVP")
    parser.add_argument("--video", default='data/raw/mini_run.mp4')
    parser.add_argument("--output", default="results")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    info = get_video_info(str(video))
    print("VIDEO:", info)

    detector = PoseDetector()
    df = detector.process_video(str(video))
    df = smooth_landmarks(df, info["fps"])

    if df.empty:
        raise RuntimeError("Бегун не обнаружен на видео.")

    df = compute_metrics(
    df,
    width=info["width"],
    height=info["height"],
)
    left_ic_raw = detect_initial_contacts(
        df,
        side="left",
        fps=info["fps"],
    )

    right_ic_raw = detect_initial_contacts(
        df,
        side="right",
        fps=info["fps"],
    )

    LEFT_IC_OFFSET = -2
    RIGHT_IC_OFFSET = -1
    left_ic = left_ic_raw + LEFT_IC_OFFSET
    left_ic = left_ic[left_ic >= 0]

    right_ic = right_ic_raw + RIGHT_IC_OFFSET
    right_ic = right_ic[right_ic >= 0]

    left_to = detect_toe_off(
        df,
        side="left",
        ic_frames=left_ic,
        fps=info["fps"],
        height=info["height"],
    )

    right_to = detect_toe_off(
        df,
        side="right",
        ic_frames=right_ic,
        fps=info["fps"],
        height=info["height"],
    )

    save_gait_events_plot(
        df,
        left_ic,
        left_to,
        "left",
        out,
    )

    save_gait_events_plot(
        df,
        right_ic,
        right_to,
        "right",
        out,
    )

    print("Left IC:", left_ic)
    print("Left TO:", left_to)

    print("Right IC:", right_ic)
    print("Right TO:", right_to)

    save_ic_plot(
    df,
    left_ic,
    "left",
    out,
    )

    save_ic_plot(
        df,
        right_ic,
        "right",
        out,
    )


    csv_path = out / "frame_metrics.csv"
    df.to_csv(csv_path, index=False)

    save_plots(df, out)
    detector.close()

    report = {
        "video": str(video),
        "fps": info["fps"],
        "left_initial_contacts": left_ic.tolist(),
        "right_initial_contacts": right_ic.tolist(),
        "frames": info["frames"],
        "duration_s": info["duration_s"],
        #"step_frequency_hz": freq,
        #"cadence_steps_min": freq * 60 if freq is not None else None,
        "rows_with_pose": int(len(df)),
        "left_toe_offs": left_to.tolist(),
        "right_toe_offs": right_to.tolist(),
    }

    import json
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_video:
        annotate_video(
            str(video),
            df,
            str(out / "annotated.mp4"),
            left_ic=left_ic,
            right_ic=right_ic,
            left_to=left_to,
            right_to=right_to
        )

    print("\nГотово.")
    print("CSV:", csv_path)
    print("REPORT:", out / "report.json")


if __name__ == "__main__":
    main()
