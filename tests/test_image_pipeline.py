"""Full image pipeline: ray-cast stereo images -> FAST/KLT front-end -> MSCKF."""
import numpy as np

from insight_vio.pipeline import run_vio_on_rendered
from insight_vio.render import RoomRenderer
from insight_vio.sim import SimConfig, simulate


def test_renderer_produces_textured_images():
    d = simulate(SimConfig(duration=0.5))
    cam = d.cameras[0]
    img = RoomRenderer().render(cam, d.gt_R[0] @ cam.R_bc, d.gt_p[0] + d.gt_R[0] @ cam.p_bc)
    assert img.shape == (480, 640) and img.dtype == np.uint8
    assert img.std() > 20


def test_image_based_stereo_vio_tracks_ground_truth():
    d = simulate(SimConfig(duration=3.0, seed=1))
    res, timing = run_vio_on_rendered(d, stereo=True)
    print(f"\n[rendered stereo VIO 3 s] ATE rmse = {res.ate.rmse * 100:.2f} cm, "
          f"front+back-end {timing['vio_s'] * 1e3:.0f} ms/frame, stats={res.stats}")
    assert res.stats["features_used"] > 200
    assert res.ate.rmse < 0.03
