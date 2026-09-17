"""
paper_mesh.py — silver nanowire mesh for the DGTD SPP simulation.
Zhao et al., PRB 113, 085425 (2026), Fig. 2 geometry.

  R = 15 nm, L = 1.2 um, hemispherical caps
  physical domain 1.8 x 0.6 x 0.6 um, PML margin 0.3 um per side
  written in micrometres (the solver scales x1000 -> nm)

Run:  python paper_mesh.py
"""
import gmsh
import sys

gmsh.initialize(sys.argv)
gmsh.model.add("Ag_Nanowire_R15")

# OCC robustness: the cylinder+sphere fuse can hit orientation warnings
gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)

# ---------------------------------------------------------------- parameters
L_wire    = 1200.0
R_wire    = 15.0
d_PhyD_x  = 1800.0
d_PhyD_yz = 600.0
d_PML     = 300.0

# Resolution. 5 nm near the wire resolves the 15 nm radius and the sigma_e=5 nm
# source; coarse far field. Raising SIZE_MIN speeds the run but under-resolves
# the metal skin.
SIZE_MIN = 5.0
SIZE_MAX = 150.0
DIST_MIN = 20.0
DIST_MAX = 250.0

# ---------------------------------------------------------------- geometry
cyl       = gmsh.model.occ.addCylinder(-L_wire/2, 0, 0, L_wire, 0, 0, R_wire)
cap_left  = gmsh.model.occ.addSphere(-L_wire/2, 0, 0, R_wire)
cap_right = gmsh.model.occ.addSphere( L_wire/2, 0, 0, R_wire)
gmsh.model.occ.fuse([(3, cyl)], [(3, cap_left), (3, cap_right)])
gmsh.model.occ.synchronize()
try:
    gmsh.model.occ.healShapes()
    gmsh.model.occ.synchronize()
except Exception:
    pass

phy_box = gmsh.model.occ.addBox(-d_PhyD_x/2, -d_PhyD_yz/2, -d_PhyD_yz/2,
                                d_PhyD_x, d_PhyD_yz, d_PhyD_yz)
tot_x  = d_PhyD_x  + 2*d_PML
tot_yz = d_PhyD_yz + 2*d_PML
total_box = gmsh.model.occ.addBox(-tot_x/2, -tot_yz/2, -tot_yz/2,
                                  tot_x, tot_yz, tot_yz)

gmsh.model.occ.fragment([(3, total_box)], [(3, phy_box), (3, cyl)])
gmsh.model.occ.synchronize()

# ------------------------------------------- find the wire volume AFTER fragment
wire_vol = None
all_vols = gmsh.model.getEntities(3)
for (dim, tag) in all_vols:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    if (ymax-ymin) < 4*R_wire and (zmax-zmin) < 4*R_wire:
        wire_vol = tag
        break
if wire_vol is None:
    raise RuntimeError("could not locate the wire volume after fragment()")

other = [t for (d, t) in all_vols if t != wire_vol]
gmsh.model.addPhysicalGroup(3, [wire_vol], tag=1, name="silver")
gmsh.model.addPhysicalGroup(3, other,      tag=2, name="vacuum_pml")

# ---------------------------------------------------------------- size field
wire_surfaces = [t for (d, t) in
                 gmsh.model.getBoundary([(3, wire_vol)], oriented=False)]
print("wire surfaces found:", wire_surfaces)
if not wire_surfaces:
    raise RuntimeError("wire surface list empty — boolean union failed")

gmsh.model.mesh.field.add("Distance", 1)
try:
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", wire_surfaces)
except Exception:
    gmsh.model.mesh.field.setNumbers(1, "FacesList", wire_surfaces)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", SIZE_MIN)
gmsh.model.mesh.field.setNumber(2, "SizeMax", SIZE_MAX)
gmsh.model.mesh.field.setNumber(2, "DistMin", DIST_MIN)
gmsh.model.mesh.field.setNumber(2, "DistMax", DIST_MAX)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", SIZE_MAX)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)

# Sliver control: the explicit timestep is set by the SMALLEST element, so
# quality optimization directly buys wall-clock time.
gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)

gmsh.model.mesh.generate(3)
# Repeated optimization passes: the explicit timestep is set by the SMALLEST
# element, so removing slivers directly buys wall-clock time. One pass is
# usually not enough on curved metal surfaces.
for _ in range(3):
    gmsh.model.mesh.optimize("Netgen")
gmsh.model.mesh.optimize("Laplace2D")

gmsh.option.setNumber("Mesh.ScalingFactor", 0.001)   # nm -> um on write
gmsh.write("paper_mesh.msh")

# report the min insphere: this is what sets the timestep
import numpy as _np
_nt, _nc, _ = gmsh.model.mesh.getNodes()
_pts = _np.array(_nc).reshape(-1,3)
_t2i = {int(t): i for i, t in enumerate(_nt)}
_et, _etag, _en = gmsh.model.mesh.getElements(dim=3)
_ti = list(_et).index(4)
_conn = _np.vectorize(_t2i.get)(_np.array(_en[_ti]).reshape(-1,4))
_v = _pts[_conn]
_a,_b,_c,_d = _v[:,0],_v[:,1],_v[:,2],_v[:,3]
_vol = _np.abs(_np.einsum('ij,ij->i', _np.cross(_b-_a,_c-_a), _d-_a))/6.0
_ar = lambda p,q,r: 0.5*_np.linalg.norm(_np.cross(q-p,r-p),axis=1)
_A = _ar(_a,_b,_c)+_ar(_a,_b,_d)+_ar(_b,_c,_d)+_ar(_a,_c,_d)
_rin = 3.0*_vol/_A
print(f"insphere radius: min {_rin.min():.3f}, 1st pct "
      f"{_np.percentile(_rin,1):.3f}, median {_np.median(_rin):.3f} nm")
print("  (min insphere sets the timestep; aim for > ~1.5 nm)")

ntet = len(gmsh.model.mesh.getElements(dim=3)[1][0])
print(f"paper_mesh.msh: R={R_wire} nm, {ntet} tetrahedra, "
      f"near-wire size ~{SIZE_MIN} nm")
gmsh.finalize()
