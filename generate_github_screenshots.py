"""Generate high-resolution 3D screenshots for GitHub documentation."""
from pathlib import Path
import vtk

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "organ_3d"
DOCS_IMG = ROOT / "docs" / "images"
DOCS_IMG.mkdir(parents=True, exist_ok=True)


def mesh_actor(filename, color, opacity):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(DATA / filename))
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(25)
    return actor


def capture_screenshot(renderer, window, filename_out, camera_pos_offset=(0, -2.2, 0.2), view_up=(0, 0, 1), zoom_factor=1.0):
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    bounds = renderer.ComputeVisiblePropBounds()
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    camera.SetFocalPoint(*center)
    camera.SetPosition(
        center[0] + camera_pos_offset[0] * size,
        center[1] + camera_pos_offset[1] * size,
        center[2] + camera_pos_offset[2] * size,
    )
    camera.SetViewUp(*view_up)
    if zoom_factor != 1.0:
        camera.Zoom(zoom_factor)
    renderer.ResetCameraClippingRange()
    window.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(window)
    w2i.SetScale(2)  # High-DPI 2x supersampling
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(DOCS_IMG / filename_out))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"Captured screenshot: {filename_out}")


def main():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.015, 0.02, 0.035)
    renderer.SetBackground2(0.08, 0.10, 0.16)
    renderer.GradientBackgroundOn()

    body = mesh_actor("body_surface.vtp", (0.72, 0.48, 0.36), 0.12)
    bone = mesh_actor("bone_surface.vtp", (0.96, 0.91, 0.78), 0.88)
    thyroid = mesh_actor("thyroid_candidate.vtp", (1.0, 0.12, 0.38), 0.90)
    thyroid_ct = mesh_actor("thyroid_ct_approx.vtp", (0.15, 0.85, 1.0), 1.0)
    heart = mesh_actor("heart_ct_approx.vtp", (0.85, 0.10, 0.18), 0.70)
    coronary = mesh_actor("coronary_calcium_candidates.vtp", (1.0, 0.85, 0.1), 1.0)
    spine = mesh_actor("cervical_spine_ct.vtp", (0.35, 0.85, 1.0), 1.0)

    for actor in (body, bone, thyroid, thyroid_ct, heart, coronary, spine):
        renderer.AddActor(actor)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1200, 900)
    window.AddRenderer(renderer)

    # 1. Full Frontal Overview
    body.SetVisibility(True)
    bone.SetVisibility(True)
    thyroid_ct.SetVisibility(True)
    heart.SetVisibility(True)
    thyroid.SetVisibility(False)
    coronary.SetVisibility(False)
    spine.SetVisibility(False)
    capture_screenshot(renderer, window, "01_full_overview_front.png", camera_pos_offset=(0, -2.0, 0))

    # 2. Oblique View
    capture_screenshot(renderer, window, "02_oblique_3d_view.png", camera_pos_offset=(1.5, -1.5, 0.6))

    # 3. Bone & Cardiovascular (Body Hidden)
    body.SetVisibility(False)
    coronary.SetVisibility(True)
    spine.SetVisibility(True)
    capture_screenshot(renderer, window, "03_bone_and_cardiovascular.png", camera_pos_offset=(-1.2, -1.6, 0.4))

    # 4. Thyroid & Cervical Spine Close-up
    thyroid.SetVisibility(True)
    capture_screenshot(renderer, window, "04_thyroid_neck_detail.png", camera_pos_offset=(0, -0.8, 0.8), zoom_factor=2.2)

    print("All screenshots successfully captured!")


if __name__ == "__main__":
    main()
