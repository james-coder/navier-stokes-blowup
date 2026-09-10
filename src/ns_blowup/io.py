"""Portable exports, independent of optional visualization packages."""
from pathlib import Path

import numpy as np


def write_vtk(path, grid, velocity):
    """Write point-sampled velocity with x fastest in the VTK stream."""
    velocity = np.asarray(velocity)
    if velocity.shape != grid.shape + (grid.ndim,):
        raise ValueError("velocity shape does not match grid")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("cannot export nonfinite velocity")
    shape = grid.shape + (1,) * (3-grid.ndim)
    spacing = grid.spacing + (1.0,) * (3-grid.ndim)
    if grid.ndim == 2:
        velocity = np.concatenate([velocity, np.zeros(grid.shape + (1,))], axis=-1)
    vectors = np.stack([velocity[..., i].ravel(order="F") for i in range(3)], axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# vtk DataFile Version 3.0\nns-blowup velocity\nASCII\nDATASET STRUCTURED_POINTS\n")
        stream.write("DIMENSIONS %d %d %d\n" % shape)
        stream.write("ORIGIN 0 0 0\nSPACING %.17g %.17g %.17g\n" % spacing)
        stream.write(f"POINT_DATA {len(vectors)}\nVECTORS velocity double\n")
        np.savetxt(stream, vectors, fmt="%.17g")
    return path
