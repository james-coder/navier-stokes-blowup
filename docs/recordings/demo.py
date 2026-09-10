from ns_blowup import Simulation
result = Simulation((32,32), nu=0.05, device="cpu").run("taylor-green", t_end=1, frames=21)
print("Velocity shape:", result.velocity.shape)
print("Device:", result.metadata["device"])
print("Final energy:", result.diagnostics()["energy"][-1])
print("NPZ:", result.save("flow.npz"))
print("VTK:", result.export_vtk("flow.vtk"))
result.plot(quantity="vorticity", path="vorticity.png")
print("Plot: vorticity.png")
