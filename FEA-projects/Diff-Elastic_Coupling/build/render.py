#!/usr/bin/env pvpython

import os
import glob
import re
from paraview.simple import *

# --------------------------------------------------
# Output directory
# --------------------------------------------------
output_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "figures"))
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Helper
# --------------------------------------------------
def extract_step(filename):
    match = re.search(r'-(\d+)\.vtu', filename)
    return int(match.group(1)) if match else -1


# --------------------------------------------------
# Create ONE render view
# --------------------------------------------------
view = CreateRenderView()
view.ViewSize = [1400, 1200]


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
    reader.UpdatePipeline()

    display = Show(reader, view)
    ColorBy(display, ('POINT_DATA', 'concentration'))
    display.RescaleTransferFunctionToDataRange(True, False)
    display.SetScalarBarVisibility(view, True)

    view.ResetCamera()

    SaveScreenshot(
        os.path.join(output_dir, f"concentration-{step}.png"),
        view
    )

    Hide(reader, view)
    Delete(reader)


# ==================================================
# 2) Displacement components
# ==================================================
disp_files = sorted(
    glob.glob("displacement-*.vtu"),
    key=extract_step
)

for file in disp_files:
    step = extract_step(file)
    print(f"Rendering displacement step {step}")

    reader = XMLUnstructuredGridReader(FileName=[file])
    reader.UpdatePipeline()

    for comp in ["u_x", "u_y"]:
        display = Show(reader, view)
        ColorBy(display, ('POINT_DATA', comp))
        display.RescaleTransferFunctionToDataRange(True, False)
        display.SetScalarBarVisibility(view, True)

        view.ResetCamera()

        SaveScreenshot(
            os.path.join(output_dir, f"{comp}-{step}.png"),
            view
        )

        Hide(reader, view)

    Delete(reader)

print("Rendering complete.")

