"""Real MediaPipe + OpenCV + FFmpeg smoke test with a no-person video."""
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from climb_app.config import Config
from climb_app.pipeline import VideoProcessor, video_metadata


def test_real_pipeline_preserves_frames_and_missing_pose(tmp_path):
    cfg = Config()
    if not cfg.model_path.is_file():
        pytest.skip('Run python setup_model.py for the real-model integration test.')
    source = tmp_path/'blank.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*cfg.codec), 30, (160, 120))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    result = VideoProcessor(cfg).process(source, tmp_path/'result')
    assert result.summary['frames'] == 30
    assert result.summary['unknown_s'] == 1
    assert result.summary['route']['color'] is None
    assert result.summary['technique']['available'] is False
    assert result.summary['coaching'][0]['code'] == 'recording'
    data = json.loads((tmp_path/'result/report.json').read_text())
    assert len(data['metrics']) == 30
    assert video_metadata(tmp_path/'result/preview.mp4', cfg)['frames'] == 30
    assert (tmp_path/'result/metrics.csv').is_file()


def test_invalid_video_is_rejected(tmp_path):
    path = tmp_path/'bad.mp4'
    path.write_text('not a video')
    with pytest.raises(ValueError):
        video_metadata(path, Config())
