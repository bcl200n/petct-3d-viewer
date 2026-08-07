"""
Create reliable 3D PET/CT visualizations from the DICOM series in this folder.

The thyroid overlay is an anatomical candidate region, not a diagnostic
segmentation. It is deliberately labeled as such in every output.
"""
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import vtk
from vtk.util.numpy_support import numpy_to_vtk


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "organ_3d"
OUT.mkdir(exist_ok=True)


def load_series(folder):
    datasets = [
        pydicom.dcmread(f, force=True)
        for f in glob.glob(str(ROOT / folder / "*.dcm"))
    ]
    datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    first = datasets[0]
    volume = np.stack([ds.pixel_array for ds in datasets]).astype(np.float32)
    volume *= float(getattr(first, "RescaleSlope", 1))
    volume += float(getattr(first, "RescaleIntercept", 0))
    z = np.array([float(ds.ImagePositionPatient[2]) for ds in datasets])
    spacing = (
        float(first.PixelSpacing[1]),
        float(first.PixelSpacing[0]),
        float(np.median(np.diff(z))),
    )
    origin = tuple(float(v) for v in first.ImagePositionPatient)
    return volume, spacing, origin, z


def vtk_image(array, spacing, origin):
    image = vtk.vtkImageData()
    image.SetDimensions(array.shape[2], array.shape[1], array.shape[0])
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    scalars = numpy_to_vtk(
        np.ascontiguousarray(array).ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    image.GetPointData().SetScalars(scalars)
    return image


def surface(image, threshold, color, opacity=1.0, smooth=10):
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, threshold)

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(smooth)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.08)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(60)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetSpecular(0.25)
    actor.GetProperty().SetSpecularPower(20)
    return actor, normals


def ellipsoid_actor(center, radii, color=(0.95, 0.2, 0.45), opacity=0.75):
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(64)
    sphere.SetPhiResolution(48)
    sphere.SetRadius(1.0)
    transform = vtk.vtkTransform()
    transform.Translate(*center)
    transform.Scale(*radii)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(transform)
    tf.SetInputConnection(sphere.GetOutputPort())
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    return actor


def text_actor(text):
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.SetPosition(24, 24)
    prop = actor.GetTextProperty()
    prop.SetFontSize(20)
    prop.SetColor(0.95, 0.95, 0.95)
    return actor


def render(actors, path, azimuth=0, elevation=5, label=""):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.015, 0.02, 0.035)
    for actor in actors:
        renderer.AddActor(actor)
    renderer.AddActor2D(text_actor(label))

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1000, 1000)
    window.AddRenderer(renderer)
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
    camera.SetPosition(center[0], center[1] - size * 2.2, center[2])
    camera.SetViewUp(0, 0, 1)
    camera.Azimuth(azimuth)
    camera.Elevation(elevation)
    camera.Zoom(1.05)
    renderer.ResetCameraClippingRange()
    window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetScale(1)
    capture.ReadFrontBufferOff()
    capture.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()


def save_polydata(connection, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(connection.GetOutputPort())
    writer.Write()


def save_actor_meshes(actors, path):
    append = vtk.vtkAppendPolyData()
    for actor in actors:
        mapper = actor.GetMapper()
        mapper.Update()
        append.AddInputData(mapper.GetInput())
    append.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(append.GetOutputPort())
    writer.Write()


def neck_montage(ct, spacing, origin, z_positions, thyroid_z):
    z_targets = np.linspace(thyroid_z - 45, thyroid_z + 45, 12)
    fig, axes = plt.subplots(3, 4, figsize=(12, 10), facecolor="#05070d")
    extent = [
        origin[0],
        origin[0] + spacing[0] * ct.shape[2],
        origin[1] + spacing[1] * ct.shape[1],
        origin[1],
    ]
    for ax, target in zip(axes.flat, z_targets):
        index = int(np.argmin(np.abs(z_positions - target)))
        ax.imshow(ct[index], cmap="gray", vmin=-150, vmax=300, extent=extent)
        ax.set_title(f"z={z_positions[index]:.0f} mm", color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        "Neck CT localization slices - thyroid must be confirmed manually",
        color="white",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "neck_ct_localization.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


print("Loading whole-body CT...")
ct, ct_spacing, ct_origin, ct_z = load_series("Data8")
ct_img = vtk_image(ct, ct_spacing, ct_origin)

# Approximate lower-neck location. This overlay is intentionally a candidate
# marker rather than a claimed segmentation.
thyroid_z = -326.0
center_x = ct_origin[0] + ct_spacing[0] * ct.shape[2] / 2
center_y = -38.0
thyroid_left = ellipsoid_actor((center_x - 14, center_y, thyroid_z), (12, 8, 18))
thyroid_right = ellipsoid_actor((center_x + 14, center_y, thyroid_z), (12, 8, 18))
thyroid_bridge = ellipsoid_actor((center_x, center_y - 1, thyroid_z - 2), (11, 5, 6))

print("Extracting body and bone surfaces...")
body_actor, body_mesh = surface(ct_img, -350, (0.72, 0.54, 0.42), opacity=0.16, smooth=6)
bone_actor, bone_mesh = surface(ct_img, 250, (0.96, 0.91, 0.78), opacity=0.92, smooth=8)
save_polydata(body_mesh, OUT / "body_surface.vtp")
save_polydata(bone_mesh, OUT / "bone_surface.vtp")

thyroid_actors = [thyroid_left, thyroid_right, thyroid_bridge]
save_actor_meshes(thyroid_actors, OUT / "thyroid_candidate.vtp")
label = "PET/CT 3D overview | Pink = thyroid candidate region (manual confirmation required)"
render([body_actor, bone_actor] + thyroid_actors, OUT / "overview_front.png", label=label)
render([body_actor, bone_actor] + thyroid_actors, OUT / "overview_oblique.png", azimuth=38, elevation=8, label=label)

print("Creating focused neck view...")
neck_ct = ct.copy()
zz = ct_origin[2] + np.arange(ct.shape[0]) * ct_spacing[2]
keep = (zz >= thyroid_z - 100) & (zz <= thyroid_z + 100)
neck_ct[~keep] = -1000
neck_img = vtk_image(neck_ct, ct_spacing, ct_origin)
neck_body, _ = surface(neck_img, -350, (0.70, 0.50, 0.40), opacity=0.12, smooth=6)
neck_bone, _ = surface(neck_img, 220, (0.95, 0.90, 0.78), opacity=0.75, smooth=8)
neck_label = "Focused neck view | Pink = approximate thyroid candidate, not diagnostic segmentation"
render([neck_body, neck_bone] + thyroid_actors, OUT / "thyroid_candidate_front.png", elevation=0, label=neck_label)
render([neck_body, neck_bone] + thyroid_actors, OUT / "thyroid_candidate_oblique.png", azimuth=35, elevation=5, label=neck_label)
neck_montage(ct, ct_spacing, ct_origin, ct_z, thyroid_z)

print(f"Done. Outputs: {OUT}")
