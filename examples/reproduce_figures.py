"""Reproduce this project's measured precision figure, not the paper's full figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ns_blowup.observatory import build_observatory


def main():
    output=Path('outputs/figures')
    output.mkdir(parents=True,exist_ok=True)
    path=build_observatory(output/'observatory.html')
    data=json.loads(path.with_suffix('.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4))
    rows=[]
    for precision in ('float32','float64'):
        measured=[r for r in data['convergence'] if r['precision']==precision]
        ax.loglog([r['dx'] for r in measured],[r['absolute_residual'] for r in measured],
                  'o-',label=precision.upper())
        for r in measured: rows.append([32 if precision=='float32' else 64,r['dx'],r['absolute_residual']])
    ax.set(xlabel='Radial stencil spacing',ylabel='Absolute heat-equation residual',
           title='Heat exterior: measured precision / refinement tradeoff')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(output/'heat-residual.png',dpi=180)
    fig.savefig(output/'heat-residual.svg')
    plt.close(fig)
    np.savetxt(output/'heat-residual.csv',rows,delimiter=',',header='precision_bits,dx,absolute_residual',comments='')
    print(output)


if __name__=='__main__': main()
