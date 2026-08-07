"""Interactive PET/CT surface viewer.

Mouse:
  Left drag = rotate, right drag = pan, wheel = zoom
Keys:
  1 = body, 2 = bone, 3 = thyroid candidate, 4 = CT thyroid, 5 = heart, 7 = neck
"""
from pathlib import Path

import vtk


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "organ_3d"


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
    actor.GetProperty().SetSpecular(0.25)
    actor.GetProperty().SetSpecularPower(20)
    return actor


renderer = vtk.vtkRenderer()
renderer.SetBackground(0.015, 0.02, 0.035)
renderer.SetBackground2(0.08, 0.10, 0.16)
renderer.GradientBackgroundOn()

body = mesh_actor("body_surface.vtp", (0.72, 0.48, 0.36), 0.12)
bone = mesh_actor("bone_surface.vtp", (0.96, 0.91, 0.78), 0.86)
thyroid = mesh_actor("thyroid_candidate.vtp", (1.0, 0.12, 0.38), 0.90)
thyroid_ct = mesh_actor("thyroid_ct_approx.vtp", (0.15, 0.85, 1.0), 1.0)
heart = mesh_actor("heart_ct_approx.vtp", (0.85, 0.10, 0.18), 0.65)
coronary_candidates = mesh_actor("coronary_calcium_candidates.vtp", (1.0, 0.85, 0.1), 1.0)
cervical_spine = mesh_actor("cervical_spine_ct.vtp", (0.35, 0.85, 1.0), 1.0)
for actor in (body, bone, thyroid, thyroid_ct, heart, coronary_candidates, cervical_spine):
    renderer.AddActor(actor)
thyroid.SetVisibility(False)
heart.SetVisibility(False)
coronary_candidates.SetVisibility(False)
cervical_spine.SetVisibility(False)

window = vtk.vtkRenderWindow()
window.SetWindowName("PET/CT Interactive 3D Viewer")
window.SetSize(1100, 850)
window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)

help_text = vtk.vtkTextActor()
help_text.SetInput(
    "Mouse: rotate / pan / zoom\n"
    "1 Body  2 Bone  3 Thyroid candidate  4 CT thyroid  5 Heart  6 Coronary  7 Cervical spine\n"
    "CT-derived models are approximate; visualization only"
)
help_text.SetPosition(20, 20)
help_text.GetTextProperty().SetFontSize(18)
help_text.GetTextProperty().SetColor(0.95, 0.95, 0.95)
renderer.AddViewProp(help_text)

title = vtk.vtkTextActor()
title.SetInput("PET/CT 3D | CT-derived approximate thyroid highlighted in cyan")
title.SetPosition(20, 805)
title.GetTextProperty().SetFontSize(22)
title.GetTextProperty().SetColor(0.85, 0.90, 1.0)
renderer.AddViewProp(title)


def reset_camera():
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
    camera.SetPosition(center[0], center[1] - size * 2.0, center[2])
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCameraClippingRange()


def on_key(_obj, _event):
    key = interactor.GetKeySym().lower()
    if key == "1":
        body.SetVisibility(not body.GetVisibility())
    elif key == "2":
        bone.SetVisibility(not bone.GetVisibility())
    elif key == "3":
        thyroid.SetVisibility(not thyroid.GetVisibility())
    elif key == "4":
        thyroid_ct.SetVisibility(not thyroid_ct.GetVisibility())
    elif key == "5":
        heart.SetVisibility(not heart.GetVisibility())
    elif key == "6":
        coronary_candidates.SetVisibility(not coronary_candidates.GetVisibility())
    elif key == "7":
        cervical_spine.SetVisibility(not cervical_spine.GetVisibility())
    elif key == "r":
        reset_camera()
    elif key in ("q", "escape"):
        window.Finalize()
        interactor.TerminateApp()
        return
    window.Render()


interactor.AddObserver("KeyPressEvent", on_key)
reset_camera()
window.Render()
interactor.Initialize()
interactor.Start()
