"""Download the official MediaPipe model once: python setup_model.py."""
from pathlib import Path
from urllib.request import urlopen
import shutil

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'


def main() -> None:
    target = Path(__file__).resolve().parent/'models'/'pose_landmarker_full.task'
    if target.is_file():
        print(f'Model already present: {target}')
        return
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.download')
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError('Model download was unexpectedly small.')
        temporary.replace(target)
        print(f'Model ready: {target}')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
