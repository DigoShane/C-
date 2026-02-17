import pyvista as pv
from pathlib import Path
import os

# ---- SETTINGS ----
DATA_FIELD = "concentration"
IMAGE_SIZE = (1000, 800)
COLORMAP = "viridis"
BACKGROUND = "white"

# ----------------------------------------------------------
# Determine output directory:
# Parent of current directory (assumes we run inside build/)
# ----------------------------------------------------------
current_dir = Path.cwd()
project_root = current_dir.parent
figures_dir = project_root / "Figures"

figures_dir.mkdir(exist_ok=True)

print(f"Saving PNGs to: {figures_dir.resolve()}\n")


def convert_vtu_to_png(vtu_file: Path):
    print(f"Rendering: {vtu_file}")

    mesh = pv.read(vtu_file)

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background(BACKGROUND)

    plotter.add_mesh(
        mesh,
        scalars=DATA_FIELD,
        cmap=COLORMAP,
        show_edges=False
    )

    plotter.view_xy()
    plotter.camera.zoom(1.2)

    # Save into Figures directory
    output_png = figures_dir / (vtu_file.stem + ".png")

    plotter.show(
        screenshot=str(output_png),
        window_size=IMAGE_SIZE
    )

    plotter.close()

    print(f"Saved: {output_png}\n")


def main():
    # Recursively find all .vtu files inside build directory
    vtu_files = list(Path(".").rglob("*.vtu"))

    if not vtu_files:
        print("No .vtu files found.")
        return

    print(f"Found {len(vtu_files)} VTU files.\n")

    for vtu_file in sorted(vtu_files):
        convert_vtu_to_png(vtu_file)


if __name__ == "__main__":
    main()

