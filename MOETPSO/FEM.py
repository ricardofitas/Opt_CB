import gmsh
import numpy as np

gmsh.initialize()

# Define the model
gmsh.model.add("corrugated_board")

# Parameters
bloc_length = 100
bloc_heigth = 15
flute_length = 40.0  # Length of the flute in mm
flute_wavelength = 5.0  # Wavelength of the flute's wave pattern in mm
flute_thickness = 0.5  # Thickness of the flute in mm
liner_thickness = 1.0  # Thickness of the liners in mm
num_flute_points = 200  # Number of points to define the flute curve
flute_amplitude = 3.0  # Amplitude of the flute's wave pattern in mm

# Create the top and bottom liner
top_bloc = gmsh.model.occ.addRectangle(-(bloc_length - flute_length)/2, -liner_thickness - flute_amplitude - bloc_heigth, 0, bloc_length, bloc_heigth)
top_liner = gmsh.model.occ.addRectangle(0, -liner_thickness - flute_amplitude, 0, flute_length, liner_thickness)
bottom_liner = gmsh.model.occ.addRectangle(0, flute_amplitude + flute_thickness, 0, flute_length, liner_thickness)
bottom_bloc = gmsh.model.occ.addRectangle(-(bloc_length - flute_length)/2, flute_amplitude + flute_thickness + liner_thickness, 0, bloc_length, bloc_heigth)

# Create points for the flute's sinusoidal curve
x_coords = np.linspace(0, flute_length, num_flute_points)
y_coords_1 = flute_amplitude * np.sin(2 * np.pi * x_coords / flute_wavelength)
y_coords_2 = flute_thickness + y_coords_1
points1 = [gmsh.model.occ.addPoint(x, y, 0) for x, y in zip(x_coords, y_coords_1)]
points2 = [gmsh.model.occ.addPoint(x, y, 0) for x, y in zip(x_coords, y_coords_2)]
# Connect the points with spline to form the flute's curve
flute_curve_1 = gmsh.model.occ.addSpline(points1)
flute_curve_2 = gmsh.model.occ.addSpline(points2[::-1])

# Create lines to connect the endpoints of the splines
line1 = gmsh.model.occ.addLine(points1[-1], points2[-1])
line2 = gmsh.model.occ.addLine(points2[0], points1[0])

# Create a curve loop from the splines and lines
loop = gmsh.model.occ.addCurveLoop([flute_curve_1, line1, flute_curve_2, line2])
flute_surface = gmsh.model.occ.addSurfaceFilling(loop)

# Extrude the first flute's curve to create the 3D shape
#flute_surface_1 = gmsh.model.occ.extrude([(1, flute_curve_1)], 0, 0, flute_thickness)

# Extrude the second flute's curve to create the 3D shape
#flute_surface_2 = gmsh.model.occ.extrude([(1, flute_curve_2)], 0, 0, flute_thickness)

# Synchronize to compile the geometry
gmsh.model.occ.synchronize()

# Mesh the combined geometry
top_bloc_group = gmsh.model.addPhysicalGroup(2, [top_bloc])
gmsh.model.setPhysicalName(3, top_bloc_group, "TopBloc")

top_liner_group = gmsh.model.addPhysicalGroup(2, [top_liner])
gmsh.model.setPhysicalName(3, top_liner_group, "TopLiner")

bottom_liner_group = gmsh.model.addPhysicalGroup(2, [bottom_liner])
gmsh.model.setPhysicalName(3, bottom_liner_group, "BottomLiner")

bottom_bloc_group = gmsh.model.addPhysicalGroup(2, [bottom_bloc])
gmsh.model.setPhysicalName(3, bottom_bloc_group, "BottomBloc")

flute_group = gmsh.model.addPhysicalGroup(2, [flute_surface])
gmsh.model.setPhysicalName(3, flute_group, "Flute")

# First, create a field that specifies the mesh size for the flute
flute_mesh_size = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(flute_mesh_size, "F", str(flute_thickness))

# Then create a Distance field that computes the distance from the flute curves
distance_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", [flute_curve_1, flute_curve_2])

# Create a Threshold field that uses the Distance field to refine the mesh near the curves
threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", flute_thickness / 2)  # Minimum mesh size near the curves
gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 10 * flute_thickness)  # Maximum mesh size away from the curves
gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 2 * flute_thickness)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 10 * flute_thickness)

# Apply this Threshold field as the background mesh field
gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)


# Now apply the mesh size field only to the flute surface using the `If` field
gmsh.model.occ.synchronize()  # Always synchronize after changing the model

# Generate the mesh
gmsh.model.mesh.generate(2)

gmsh.fltk.run()

# Save the mesh to a file
gmsh.write("corrugated_board.msh")

gmsh.finalize()