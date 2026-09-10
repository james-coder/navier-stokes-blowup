"""Run from the repo after installing: python examples/quickstart.py."""
from ns_blowup import Simulation

result = Simulation((32, 32), nu=0.05).run(t_end=1.0, frames=21)
result.save("outputs/quickstart/flow.npz")
result.plot(quantity="vorticity", path="outputs/quickstart/vorticity.png")
print({name: values[-1] for name, values in result.diagnostics().items()})
