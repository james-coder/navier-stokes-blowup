"""Infer a forcing amplitude through the numerical evolution; no global optimum claim."""
import jax
import jax.numpy as jnp
from ns_blowup import PeriodicGrid, SpectralSolver


def main():
    grid = PeriodicGrid((16,16))
    solver = SpectralSolver(grid, nu=.1)
    initial = jnp.zeros(grid.shape+(2,))

    def predict(amplitude):
        def force(x,t):
            return jnp.stack([amplitude*jnp.sin(x[...,1]),jnp.zeros_like(x[...,0])],axis=-1)
        return solver.integrate(initial,.02,25,forcing=force)

    target = predict(1.5)
    def objective(amplitude):
        return jnp.mean((predict(amplitude)-target)**2)

    loss_grad = jax.jit(jax.value_and_grad(objective))
    amplitude = jnp.asarray(.2)
    initial_loss = float(objective(amplitude))
    for _ in range(100):
        loss, gradient = loss_grad(amplitude)
        amplitude = amplitude - 1.5*gradient
    final_loss = float(objective(amplitude))
    print(f"amplitude={float(amplitude):.6f}; loss {initial_loss:.6g} -> {final_loss:.6g}")
    if final_loss >= initial_loss * 1e-6:
        raise RuntimeError("optimization did not reach the example tolerance")


if __name__ == '__main__':
    main()
