"""Download the public MIT-licensed test clips used by the application."""
from pathlib import Path
from urllib.request import urlopen
import shutil

BASE = 'https://raw.githubusercontent.com/tommyjtl/climbing-analysis-toolbox/main/examples/videos/'
NAMES = ('body-trajectory-input.mp4', 'warp-dynamic-input.mp4', 'warp-fixed-input.mp4')


def main() -> None:
    folder = Path(__file__).resolve().parent/'samples'
    folder.mkdir(exist_ok=True)
    for name in NAMES:
        target = folder/name
        if target.is_file():
            print(f'Already present: {target}')
            continue
        temporary = target.with_suffix('.download')
        try:
            print(f'Downloading {name} ...')
            with urlopen(BASE+name, timeout=60) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size < 100_000:
                raise RuntimeError(f'Download was unexpectedly small: {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print('Sample clips are ready.')


if __name__ == '__main__':
    main()
