"""Navier–Stokes simulation, exact benchmarks and construction research in JAX."""
from .analytical import ABCFlow, TaylorGreen
from .benchmarks import TaylorGreen3D
from .construction import SimilarityCoordinates, ViscosityScaled
from .diagnostics import FieldDiagnostics, kinetic_energy, sampled_norms
from .exterior import HeatExterior
from .grid import PeriodicGrid
from .simulation import Simulation, SimulationResult
from .spectral import SpectralSolver

__version__ = "0.2.0"
__all__ = ["HeatExterior", "ABCFlow", "TaylorGreen", "TaylorGreen3D", "SimilarityCoordinates", "ViscosityScaled",
           "FieldDiagnostics", "kinetic_energy", "sampled_norms", "PeriodicGrid",
           "Simulation", "SimulationResult", "SpectralSolver"]
