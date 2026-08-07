"""Extract an approximate cervical-spine model from whole-body CT."""
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


def load_ct():
    dss = [pydicom.dcmread(f, force=True) for f in glob.glob(str(ROOT / "Data8" / "*.dcm"))]
    dss.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    first = dss[0]
    ct = np.stack([d.pixel_array for d in dss]).astype(np.float32)
    ct = ct * float(first.RescaleSlope) + float(first.RescaleIntercept)
    origin = np.array([float(v) for v in first.ImagePositionPatient])
    spacing = np.array([
        float(first.PixelSpacing[1]),
        float(first.PixelSpacing[0]),
        np.median(np.diff([float(d.ImagePositionPatient[2]) for d in dss])),
    ])
    return ct, origin, spacing


def ellipsoid(shape, center_xyz, radii_xyz):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cx = (center_xyz[0] - origin[0]) / spacing[0]
    cy = (center_xyz[1] - origin[1]) / spacing[1]
    cz = (center_xyz[2] - origin[2]) / spacing[2]
    rx, ry, rz = np.array(radii_xyz) / spacing
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1


def vtk_image(mask):
    image = vtk.vtkImageData()
    image.SetDimensions(mask.shape[2], mask.shape[1], mask.shape[0])
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    scalars = numpy_to_vtk(
        np.ascontiguousarray(mask.astype(np.uint8) * 255).ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    image.GetPointData().SetScalars(scalars)
    return image


ct, origin, spacing = load_ct()

# The cervical spine lies posterior to the airway. The envelope excludes most
# skull, mandible, shoulder girdle and ribs while retaining C1-C7.
envelope = ellipsoid(ct.shape, (-4, 42, -290), (43, 48, 72))
bone = envelope & (ct >= 180)

image = vtk_image(bone)
contour = vtk.vtkFlyingEdges3D()
contour.SetInputData(image)
contour.SetValue(0, 127)

smooth = vtk.vtkWindowedSincPolyDataFilter()
smooth.SetInputConnection(contour.GetOutputPort())
smooth.SetNumberOfIterations(12)
smooth.SetPassBand(0.08)
smooth.BoundarySmoothingOff()
smooth.Update()

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(str(OUT / "cervical_spine_ct.vtp"))
writer.SetInputConnection(smooth.GetOutputPort())
writer.Write()

stl = vtk.vtkSTLWriter()
stl.SetFileName(str(OUT / "cervical_spine_ct.stl"))
stl.SetInputConnection(smooth.GetOutputPort())
stl.Write()

# Standalone preview.
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(smooth.GetOutputPort())
mapper.ScalarVisibilityOff()
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.92, 0.88, 0.76)
actor.GetProperty().SetSpecular(0.3)
actor.GetProperty().SetSpecularPower(25)
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.015, 0.02, 0.035)
renderer.AddActor(actor)
window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetSize(900, 1000)
window.AddRenderer(renderer)
renderer.ResetCamera()
renderer.GetActiveCamera().Azimuth(28)
renderer.GetActiveCamera().Elevation(5)
renderer.ResetCameraClippingRange()
window.Render()
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(window)
capture.ReadFrontBufferOff()
capture.Update()
png = vtk.vtkPNGWriter()
png.SetFileName(str(OUT / "cervical_spine_3d.png"))
png.SetInputConnection(capture.GetOutputPort())
png.Write()

# Axial validation montage.
fig, axes = plt.subplots(3, 4, figsize=(12, 10), facecolor="#05070d")
z_targets = np.linspace(-357, -225, 12)
extent = [
    origin[0],
    origin[0] + spacing[0] * ct.shape[2],
    origin[1] + spacing[1] * ct.shape[1],
    origin[1],
]
x_coords = origin[0] + np.arange(ct.shape[2]) * spacing[0]
y_coords = origin[1] + np.arange(ct.shape[1]) * spacing[1]
for ax, z_mm in zip(axes.flat, z_targets):
    k = int(round((z_mm - origin[2]) / spacing[2]))
    actual_z = origin[2] + k * spacing[2]
    ax.imshow(ct[k], cmap="gray", vmin=-200, vmax=900, extent=extent)
    if bone[k].any():
        ax.contour(x_coords, y_coords, bone[k].astype(float), levels=[0.5], colors=["#40d9ff"], linewidths=1.2)
    ax.set_xlim(-95, 95)
    ax.set_ylim(135, -100)
    ax.set_title(f"z={actual_z:.0f} mm", color="white", fontsize=9)
    ax.axis("off")
fig.suptitle("Cervical-spine extraction validation | cyan=selected bone", color="white", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "cervical_spine_validation.png", dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)

poly = smooth.GetOutput()
report = f"""颈椎CT三维模型

来源：Data8 全身CT
分辨率：{spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm
网格点数：{poly.GetNumberOfPoints()}
网格面数：{poly.GetNumberOfPolys()}

限制：
- 模型用于显示颈椎骨性结构大致外形。
- 3 mm层厚会损失细小骨折、椎间盘、神经和韧带细节。
- 不能代替颈椎薄层CT、MRI或医生诊断。
"""
(OUT / "cervical_spine_README.txt").write_text(report, encoding="utf-8")
print(report)
