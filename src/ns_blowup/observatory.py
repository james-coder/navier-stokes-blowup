"""Build the offline Singularity Observatory from computed construction components."""
import argparse
from importlib.resources import files
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .construction import SimilarityCoordinates
from .exterior import HeatExterior


def build_observatory(output="outputs/observatory.html", *, h=.005):
    """Write standalone HTML and accompanying JSON with auditable numerical data.

    This component explorer is not a simulation of the completed blowup field.
    Construction calculations prefer CPU (falling back to the enabled device), leaving
    the application's global precision setting unchanged on return.
    """
    try:
        device = jax.devices('cpu')[0]
    except RuntimeError:
        device = jax.devices()[0]
    with jax.default_device(device), jax.enable_x64(True):
        geometry=SimilarityCoordinates(h)
        tau=np.geomspace(1.,1e-12,121)
        scales={k:np.asarray(v).tolist() for k,v in geometry.scales(jnp.asarray(tau)).items()}
        eta=np.linspace(-.8,.8,161)
        boundaries=[]
        for remaining in tau:
            q=remaining/(1-eta**2)
            boundaries.append({'r':np.sqrt(2*q).tolist(),'z':(q**(.5-h)*eta).tolist()})
        exterior=HeatExterior(h,order=96)
        Z=np.geomspace(1e-5,4.,128)
        factors=np.asarray(exterior.factor(jnp.asarray(Z))).tolist()
    convergence=[]
    for precision in ('float32','float64'):
        with jax.default_device(device), jax.enable_x64(precision=='float64'):
            exterior=HeatExterior(h,order=96)
            dtype=getattr(jnp,precision)
            radius,tau0=jnp.asarray(1.5,dtype),jnp.asarray(.2,dtype)
            k=exterior.azimuthal_velocity
            kt=-jax.grad(k,1)(radius,tau0)
            for n in (8,16,32,64,128,256,512,1024):
                dx=jnp.asarray(1/(n-1),dtype)
                f0,fp,fm=k(radius,tau0),k(radius+dx,tau0),k(radius-dx,tau0)
                spatial=(fp-2*f0+fm)/dx**2+(fp-fm)/(2*dx*radius)-f0/radius**2
                absolute=float(jnp.abs(kt-spatial))
                relative=absolute/max(abs(float(kt)),1e-30)
                convergence.append(dict(precision=precision,n=n,dx=float(dx),absolute_residual=absolute,
                                        relative_residual=relative,reference_time_derivative=float(kt)))
    payload={'h':h,'tau':tau.tolist(),'scales':scales,'boundaries':boundaries,
             'heat_factor':{'Z':Z.tolist(),'H':factors},'convergence':convergence,
             'jax_version':jax.__version__,'quadrature_order':96,'compute_device':str(device),
             'status':'Component explorer. Full smooth blowup solution NOT IMPLEMENTED.',
             'reference':'OpenAI, Finite time blowup for Navier–Stokes, (3.2), Lemma A.6'}
    encoded=json.dumps(payload,allow_nan=False)
    template=files('ns_blowup').joinpath('assets/observatory.html').read_text()
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template.replace('__DATA__',encoded.replace('<','\\u003c')))
    output.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/observatory.html')
    parser.add_argument('--h',type=float,default=.005)
    args=parser.parse_args()
    print(build_observatory(args.output,h=args.h))


if __name__=='__main__': main()
