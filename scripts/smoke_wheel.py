"""Smoke-check an installed wheel from outside the source checkout."""
from importlib.resources import files
from pathlib import Path
import tempfile

import ns_blowup
from ns_blowup import Simulation
from ns_blowup.observatory import build_observatory


def main():
    package_path=Path(ns_blowup.__file__).resolve()
    if 'site-packages' not in str(package_path):
        raise RuntimeError(f'Expected an installed package, found {package_path}')
    result=Simulation((8,8),device='cpu').run(t_end=.01,frames=2)
    assert result.final.shape==(8,8,2)
    assert files('ns_blowup').joinpath('assets/observatory.html').is_file()
    with tempfile.TemporaryDirectory() as directory:
        path=build_observatory(Path(directory)/'observatory.html')
        assert path.is_file()
    print(f'Installed-wheel smoke check passed: {ns_blowup.__version__}')


if __name__=='__main__': main()
