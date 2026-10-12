def test_project_imports():
    from src.video.loader import get_video_info
    from src.biomechanics.metrics import compute_metrics
    from src.biomechanics.gait_cycle import estimate_step_frequency
    assert callable(get_video_info)
    assert callable(compute_metrics)
    assert callable(estimate_step_frequency)
