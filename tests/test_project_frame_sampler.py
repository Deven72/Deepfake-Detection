from pathlib import Path

import pytest

from backend.services.media.frame_sampler import sample_frames


def test_sampler_rejects_invalid_sampling_rate():
    with pytest.raises(ValueError):
        sample_frames("not-a-video.mp4", sample_fps=0)


def test_sampler_rejects_zero_max_frames():
    with pytest.raises(ValueError):
        sample_frames("not-a-video.mp4", max_frames=0)


def test_sampler_on_bundled_video():
    video = Path(__file__).resolve().parents[1] / "datasets" / "test_video.mp4"
    if not video.exists():
        pytest.skip("Bundled smoke-test video is not present")
    frames = sample_frames(str(video), sample_fps=1, max_frames=8)
    assert 1 <= len(frames) <= 8
    assert frames[0].ndim == 3
