#!/usr/bin/env python3

import os
import glob
import re
from paraview.simple import *

# --------------------------------------------------
# Output directory: parent folder /figures
# --------------------------------------------------
output_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "figures"))
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Helper: extract timestep
# --------------------------------------------------
def extract_step(filename):
    match = re.search(r'-(\d+)\.vtu', filename)
    return int(match.group(1)) if match else -1


# ==================================================
# 1) Concentration
# ==================================================
concentration_files = sorted(
    glob.glob("concentration-*.vtu"),
    key=extract_step
)

for file in concentration_files:
    step = extract_step(file)
    print(f"Rendering concentration step {step}")

    reader = XMLUnstructuredGridReader(FileName=[file])
    view = CreateRenderView()

    display = Show(reader, view)
    ColorBy(display, ('POINTS', 'concentration'))

    display.RescaleTransferFunctionToDataRange(True)
    display.SetScalarBarVisibility(view, True)

    view.ResetCamera()
    view.ViewSize = [1400, 1200]

    SaveScreenshot(
        os.path.join(output_dir, f"concentration-{step}.png"),
        view
    )

    Delete(reader)
    Delete(view)


# ==================================================
# 2) Displacement magnitude
# ==================================================
displacement_files = sorted(
    glob.glob("displacement-*.vtu"),
    key=extract_step
)

for file in displacement_files:
    step = extract_step(file)
    print(f"Rendering displacement step {step}")

    reader = XMLUnstructuredGridReader(FileName=[file])
    view = CreateRenderView()

    display = Show(reader, view)

    # VERY IMPORTANT:
    # Use magnitude of vector automatically
    ColorBy(display, ('POINTS', 'displacement', 'Magnitude'))

    display.RescaleTransferFunctionToDataRange(True)
    display.SetScalarBarVisibility(view, True)

    view.ResetCamera()
    view.ViewSize = [1400, 1200]

    SaveScreenshot(
        os.path.join(output_dir, f"displacement-{step}.png"),
        view
    )

    Delete(reader)
    Delete(view)

print("Rendering complete.")

