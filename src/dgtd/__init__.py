"""
dgtd — high-order nodal discontinuous Galerkin time-domain Maxwell solver
with Drude dispersive media, for electron-beam (EELS) excitation of surface
plasmon polaritons on metallic nanowires.
"""
__version__ = "0.1.0"

from .dg_nodal3d import build_reference_full
from .dg_geom import geometric_factors, normals_surface
from .dg_connect import build_face_connectivity

__all__ = [
    "build_reference_full",
    "geometric_factors",
    "normals_surface",
    "build_face_connectivity",
]
