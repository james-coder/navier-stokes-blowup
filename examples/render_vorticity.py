"""Generate the README's actual first-simulation output, from the repository root."""
import argparse
from pathlib import Path

from ns_blowup import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/images/vorticity.png'))
    args = parser.parse_args()
    result = Simulation((32, 32), nu=0.05, device='cpu').run(
        initial='taylor-green', t_end=1.0, frames=21,
    )
    result.plot(quantity='vorticity', path=args.output)
    print(f'Saved {args.output}: 32×32 Taylor–Green vorticity, nu=0.05, t=1.0')


if __name__ == '__main__':
    main()
