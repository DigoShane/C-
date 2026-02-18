#!/usr/bin/env python3

import os
import glob
import pyvista as pv

project_dir = os.path.dirname(os.path.abspath(__file__))

# Search everywhere in project
files = glob.glob(os.path.join(project_dir, "**/*.pvtu"), recursive=True)

if not files:
    files = glob.glob(os.path.join(project_dir, "**/*.vtu"), recursive=True)

if not files:
    print("❌ No VTU or PVTU files found.")
    exit()

# Create Figures folder
figure_dir = os.path.join(project_dir, "Figures")
os.makedirs(figure_dir, exist_ok=True)

def extract_number(filename):
    digits = ''.join(filter(str.isdigit, filename))
    return int(digits) if digits else 0

files.sort(key=extract_number)

print(f"Found {len(files)} files")

pv.start_xvfb()

for i, file in enumerate(files):
    print(f"Rendering {file}")

    mesh = pv.read(file)

    plotter = pv.Plotter(off_screen=True)

    # For Step-66
    if "solution" in mesh.array_names:
        scalar_name = "solution"
    elif "c" in mesh.array_names:
        scalar_name = "c"
    else:
        scalar_name = mesh.array_names[0]

    plotter.add_mesh(mesh, scalars=scalar_name, cmap="viridis")

    plotter.view_xy()

    out = os.path.join(figure_dir, f"frame-{i:04d}.png")
    plotter.show(screenshot=out)
    plotter.close()

print("✅ Done. Images saved in Figures/")

