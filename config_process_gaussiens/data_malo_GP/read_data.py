import pyvista as pv
import numpy as np

# Script to read binary files + visualize unstructured data

# =========== Parameters ===========

data_path = "/Users/blombard/Documents/config_process_gaussiens/data_malo_GP"

hs = 1
msl = 0.2

# =========== Read the data ===========

hs_list = np.arange(1, 3.1, 0.5)
msl_list = np.arange(0, 1.1, 0.1)

if hs not in hs_list :
   raise ValueError(f"Invalid value : hs = {hs}")
if msl not in msl_list :
   raise ValueError(f"Invalid value : msl = {msl}")

# Load the cell info binary file
cell_info = np.fromfile(data_path + "/scattered_land_data.bin", dtype='>f8').reshape(-1, 5)
id_gmsh = cell_info[:, 0].astype(int)-1                                                         # Convert to 0 based array

# Load the flood map binary file
fname = data_path + f"/run_msl_{msl:.1f}_hs_{hs:.1f}" + "/scattered_map_final.bin"
data = np.fromfile(fname, dtype='>f8').reshape(-1, 3)

# ===========  Data extracted from files ===========

x = cell_info[:, 1]
y = cell_info[:, 2]
bathy = cell_info[:, 3]
surf = cell_info[:, 4]
Hmax_t = data[:, 1]
Hmean_t = data[:, 2]

# =========== Visualize scattered data on mesh ===========

mesh = pv.read(data_path + "/gmsh_mapping.vtk")

# vtk_2_gmsh[id_vtk] <--> id_gmsh                gmsh_2_vtk[id_gmsh] <--> id_vtk
vtk_2_gmsh = mesh.cell_data["gmsh_cell_id"].astype(int)-1            # Convert to 0 based array
gmsh_2_vtk = np.empty_like(vtk_2_gmsh)
gmsh_2_vtk[vtk_2_gmsh] = np.arange(len(vtk_2_gmsh))
id_vtk = gmsh_2_vtk[id_gmsh]

# Data to plot
map = np.full(mesh.n_cells, np.nan)
map[id_vtk] = Hmax_t
mesh.cell_data["H"] = map

submesh = mesh.extract_cells(id_vtk)

plotter = pv.Plotter()
plotter.add_mesh(submesh, scalars="H", clim=[0,1], cmap="coolwarm")
plotter.view_xy()
plotter.show()

# =========== Project flood map on structured grid ===========

# N = 1000

# xmin, xmax, ymin, ymax = submesh.GetBounds()[0:4]
# x = np.linspace(xmin, xmax, N)
# y = np.linspace(ymin, ymax, N)

# X, Y = np.meshgrid(x, y, indexing='xy')
# sample_points = np.column_stack((X.ravel(), Y.ravel(), np.zeros(N**2)))
# cid = submesh.find_containing_cell(sample_points)

# Hmax_structured = np.full(N**2, np.nan)
# Hmax_structured[cid >= 0] = submesh.cell_data["H"][cid[cid >= 0]]
# Hmax_structured[Hmax_structured < 0] = np.nan

# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(Hmax_structured.reshape(N,N), origin='lower')
# plt.axis('off')
# plt.colorbar()
# plt.clim(0,0.5)
# plt.show()