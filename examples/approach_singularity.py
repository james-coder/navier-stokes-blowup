"""Export normalized construction scales, not a full blowup trajectory."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ns_blowup import SimilarityCoordinates


def main():
    with jax.enable_x64(True):
        tau=jnp.geomspace(1.,1e-12,121)
        scales=SimilarityCoordinates().scales(tau)
        path=Path('outputs/construction-scales.csv')
        path.parent.mkdir(parents=True,exist_ok=True)
        names=['tau',*scales]
        values=np.column_stack([np.asarray(tau),*[np.asarray(v) for v in scales.values()]])
        np.savetxt(path,values,delimiter=',',header=','.join(names),comments='')
    print(f'{path}: normalized scaling laws only; full blowup trajectory is not implemented')


if __name__=='__main__': main()
