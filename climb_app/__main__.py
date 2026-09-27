"""CLI: python -m climb_app input.mp4 --output output/my-climb"""
import argparse
from pathlib import Path

from .config import Config
from .pipeline import VideoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Config.model_path)
    parser.add_argument('--crop', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'))
    parser.add_argument('--static-speed', type=float, default=Config.static_speed)
    parser.add_argument('--smoothing', type=float, default=Config.smoothing_seconds)
    args = parser.parse_args()
    cfg = Config(model_path=args.model, static_speed=args.static_speed, smoothing_seconds=args.smoothing)
    try:
        result = VideoProcessor(cfg).process(args.input, args.output, tuple(args.crop) if args.crop else None,
                                             lambda stage, p: print(f'\r{stage}: {p:.0%}', end='', flush=True))
        print('\n', result.summary)
        for warning in result.warnings:
            print('NOTE:', warning)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'\nError: {error}\n')


if __name__ == '__main__':
    main()
