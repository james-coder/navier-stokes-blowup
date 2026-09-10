"""Reject a release tag that differs from the package version."""
import ast
import os
from pathlib import Path
import re
import tomllib


def main():
    version = tomllib.loads(Path('pyproject.toml').read_text())['project']['version']
    for filename, name in [('src/ns_blowup/__init__.py', '__version__'), ('docs/conf.py', 'release')]:
        tree = ast.parse(Path(filename).read_text())
        declared = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
        if declared != version:
            raise SystemExit(f'{filename}: {declared!r} differs from package version {version!r}')
    tag = os.environ.get('RELEASE_TAG', '')
    if not re.fullmatch(r'v[0-9][0-9A-Za-z.+-]*', tag) or tag != f'v{version}':
        raise SystemExit(f'Release tag {tag!r} must match package version v{version}')
    print(f'Release version verified: {tag}')


if __name__ == '__main__':
    main()
