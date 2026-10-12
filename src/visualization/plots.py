import matplotlib.pyplot as plt


def save_plots(df, out_dir):
    # ==========================================
    # 1. Knee angles
    # ==========================================
    plt.figure(figsize=(10, 4))
    plt.plot(
        df.frame,
        df.left_knee_angle_smooth,
        label="Left knee",
    )
    plt.plot(
        df.frame,
        df.right_knee_angle_smooth,
        label="Right knee",
    )

    plt.xlabel("Frame")
    plt.ylabel("Knee angle (deg)")
    plt.title("Knee angles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "knee_angles.png",
        dpi=160,
    )
    plt.close()

    # ==========================================
    # 2. Trunk inclination
    # ==========================================
    plt.figure(figsize=(10, 4))
    plt.plot(
        df.frame,
        df.trunk_inclination_deg,
    )

    plt.xlabel("Frame")
    plt.ylabel("Trunk inclination (deg)")
    plt.title("Trunk inclination")
    plt.tight_layout()
    plt.savefig(
        out_dir / "trunk_inclination.png",
        dpi=160,
    )
    plt.close()

    # ==========================================
    # 3. Pelvis vertical position
    # ==========================================
    plt.figure(figsize=(10, 4))
    plt.plot(
        df.frame,
        df.pelvis_y,
    )

    plt.xlabel("Frame")
    plt.ylabel("Pelvis vertical position")
    plt.title("Pelvis vertical position")
    plt.tight_layout()
    plt.savefig(
        out_dir / "pelvis_vertical.png",
        dpi=160,
    )
    plt.close()

    # ==========================================
    # 4. RAW vs SMOOTH — left knee X
    # ==========================================
    plt.figure(figsize=(10, 4))

    plt.plot(
        df.frame,
        df.left_knee_x,
        label="Raw",
    )

    plt.plot(
        df.frame,
        df.left_knee_smooth_x,
        label="Smooth",
    )

    plt.xlabel("Frame")
    plt.ylabel("Normalized X")
    plt.title("Left knee X: RAW vs SMOOTH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "left_knee_x_raw_vs_smooth.png",
        dpi=160,
    )
    plt.close()

    # ==========================================
    # 5. RAW vs SMOOTH — left ankle Y
    # ==========================================
    plt.figure(figsize=(10, 4))

    plt.plot(
        df.frame,
        df.left_ankle_y,
        label="Raw",
    )

    plt.plot(
        df.frame,
        df.left_ankle_smooth_y,
        label="Smooth",
    )

    plt.xlabel("Frame")
    plt.ylabel("Normalized Y")
    plt.title("Left ankle Y: RAW vs SMOOTH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "left_ankle_y_raw_vs_smooth.png",
        dpi=160,
    )
    plt.close()

    # ==========================================
    # 6. RAW vs SMOOTH — right knee X
    # ==========================================
    plt.figure(figsize=(10, 4))

    plt.plot(
        df.frame,
        df.right_knee_x,
        label="Raw",
    )

    plt.plot(
        df.frame,
        df.right_knee_smooth_x,
        label="Smooth",
    )

    plt.xlabel("Frame")
    plt.ylabel("Normalized X")
    plt.title("Right knee X: RAW vs SMOOTH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "right_knee_x_raw_vs_smooth.png",
        dpi=160,
    )
    plt.close()

    # ==========================================
    # 7. RAW vs SMOOTH — right ankle Y
    # ==========================================
    plt.figure(figsize=(10, 4))

    plt.plot(
        df.frame,
        df.right_ankle_y,
        label="Raw",
    )

    plt.plot(
        df.frame,
        df.right_ankle_smooth_y,
        label="Smooth",
    )

    plt.xlabel("Frame")
    plt.ylabel("Normalized Y")
    plt.title("Right ankle Y: RAW vs SMOOTH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "right_ankle_y_raw_vs_smooth.png",
        dpi=160,
    )
    plt.close()

def save_ic_plot(df, ic_frames, side, out_dir):
    import numpy as np

    toe_y = df[
        f"{side}_foot_index_smooth_y"
    ].to_numpy(dtype=float)

    heel_y = df[
        f"{side}_heel_smooth_y"
    ].to_numpy(dtype=float)

    foot_y = np.maximum(toe_y, heel_y)

    plt.figure(figsize=(12, 5))

    plt.plot(
        df.frame,
        toe_y,
        label="Toe",
    )

    plt.plot(
        df.frame,
        heel_y,
        label="Heel",
    )

    plt.plot(
        df.frame,
        foot_y,
        label="Foot envelope",
        linewidth=2,
    )

    for frame in ic_frames:
        plt.axvline(
            frame,
            linestyle="--",
            linewidth=1,
            label="Initial Contact"
            if frame == ic_frames[0]
            else None,
        )

    plt.xlabel("Frame")
    plt.ylabel("Normalized Y")
    plt.title(
        f"{side.capitalize()} Initial Contact candidates"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out_dir / f"{side}_initial_contacts.png",
        dpi=160,
    )

    plt.close()

def save_gait_events_plot(
    df,
    ic_frames,
    to_frames,
    side,
    out_dir,
):
    import numpy as np
    import matplotlib.pyplot as plt

    frames = df["frame"].to_numpy(dtype=int)

    toe_y = df[
        f"{side}_foot_index_smooth_y"
    ].to_numpy(dtype=float)

    heel_y = df[
        f"{side}_heel_smooth_y"
    ].to_numpy(dtype=float)

    foot_y = np.maximum(toe_y, heel_y)

    plt.figure(figsize=(12, 5))

    plt.plot(
        frames,
        toe_y,
        label="Toe Y",
    )

    plt.plot(
        frames,
        heel_y,
        label="Heel Y",
    )

    plt.plot(
        frames,
        foot_y,
        label="Foot envelope",
        linewidth=2,
    )

    for i, frame in enumerate(ic_frames):
        plt.axvline(
            frame,
            linestyle="--",
            alpha=0.8,
            label="IC" if i == 0 else None,
        )

    for i, frame in enumerate(to_frames):
        plt.axvline(
            frame,
            linestyle=":",
            linewidth=2,
            alpha=0.9,
            label="TO candidate" if i == 0 else None,
        )

    plt.xlabel("Frame")
    plt.ylabel("Normalized image Y (larger = lower)")
    plt.title(f"{side.capitalize()} gait events: IC and TO")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out_dir / f"{side}_gait_events.png",
        dpi=160,
    )

    plt.close()