"""Approximate heart envelope and coronary calcification candidates.

Built from non-contrast, non-ECG-gated whole-body CT. Visualization only.
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


def load_ct(folder="Data8"):
    dss = [pydicom.dcmread(f, force=True) for f in glob.glob(str(ROOT / folder / "*.dcm"))]
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


def close3d(mask, iterations=1):
    out = mask.copy()
    for _ in range(iterations):
        p = np.pad(out, 1)
        out = np.logical_or.reduce(
            [p[dz : dz + mask.shape[0], dy : dy + mask.shape[1], dx : dx + mask.shape[2]]
             for dz in range(3) for dy in range(3) for dx in range(3)]
        )
    for _ in range(iterations):
        p = np.pad(out, 1, constant_values=True)
        out = np.logical_and.reduce(
            [p[dz : dz + mask.shape[0], dy : dy + mask.shape[1], dx : dx + mask.shape[2]]
             for dz in range(3) for dy in range(3) for dx in range(3)]
        )
    return out


def vtk_image(mask):
    img = vtk.vtkImageData()
    img.SetDimensions(mask.shape[2], mask.shape[1], mask.shape[0])
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)
    scalars = numpy_to_vtk(
        np.ascontiguousarray(mask.astype(np.uint8) * 255).ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    img.GetPointData().SetScalars(scalars)
    return img


def mesh(mask, filename, smooth_iterations=15, gaussian=True):
    img = vtk_image(mask)
    contour = vtk.vtkFlyingEdges3D()
    if gaussian:
        blur = vtk.vtkImageGaussianSmooth()
        blur.SetInputData(img)
        blur.SetStandardDeviations(1.0, 1.0, 0.7)
        blur.SetRadiusFactors(2, 2, 2)
        contour.SetInputConnection(blur.GetOutputPort())
        contour.SetValue(0, 100)
    else:
        contour.SetInputData(img)
        contour.SetValue(0, 127)
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(contour.GetOutputPort())
    smooth.SetNumberOfIterations(smooth_iterations)
    smooth.SetPassBand(0.08)
    smooth.BoundarySmoothingOff()
    smooth.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(OUT / f"{filename}.vtp"))
    writer.SetInputConnection(smooth.GetOutputPort())
    writer.Write()
    stl = vtk.vtkSTLWriter()
    stl.SetFileName(str(OUT / f"{filename}.stl"))
    stl.SetInputConnection(smooth.GetOutputPort())
    stl.Write()
    return smooth


def actor(connection, color, opacity=1.0):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(connection.GetOutputPort())
    mapper.ScalarVisibilityOff()
    out = vtk.vtkActor()
    out.SetMapper(mapper)
    out.GetProperty().SetColor(*color)
    out.GetProperty().SetOpacity(opacity)
    out.GetProperty().SetSpecular(0.25)
    return out


ct, origin, spacing = load_ct()

# Broad patient-specific cardiac silhouette envelope. It follows visible
# soft-tissue density inside a conservative anatomical ellipsoid.
heart_roi = ellipsoid(ct.shape, (-12, -30, -478), (82, 55, 58))
heart = heart_roi & (ct > -70) & (ct < 250)
heart = close3d(heart, iterations=1)

# High-density candidates restricted to the anterior cardiac surface where
# coronary arteries commonly course. This excludes most spine/rib calcium,
# but may still include valves or motion artefact.
coronary_roi = ellipsoid(ct.shape, (-10, -38, -482), (68, 30, 52))
coronary_calcium = coronary_roi & (ct >= 180)

heart_mesh = mesh(heart, "heart_ct_approx", smooth_iterations=20)
calcium_mesh = mesh(coronary_calcium, "coronary_calcium_candidates", smooth_iterations=3, gaussian=False)

# Render combined preview.
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.015, 0.02, 0.035)
renderer.AddActor(actor(heart_mesh, (0.75, 0.16, 0.22), 0.35))
renderer.AddActor(actor(calcium_mesh, (1.0, 0.85, 0.18), 1.0))
window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetSize(1000, 850)
window.AddRenderer(renderer)
renderer.ResetCamera()
renderer.GetActiveCamera().Azimuth(25)
renderer.GetActiveCamera().Elevation(8)
renderer.ResetCameraClippingRange()
window.Render()
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(window)
capture.ReadFrontBufferOff()
capture.Update()
png = vtk.vtkPNGWriter()
png.SetFileName(str(OUT / "heart_coronary_3d.png"))
png.SetInputConnection(capture.GetOutputPort())
png.Write()

# Axial validation montage.
fig, axes = plt.subplots(3, 4, figsize=(12, 10), facecolor="#05070d")
z_targets = np.linspace(-540, -435, 12)
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
    ax.imshow(ct[k], cmap="gray", vmin=-150, vmax=400, extent=extent)
    ax.contour(x_coords, y_coords, heart[k].astype(float), levels=[0.5], colors=["#ff4060"], linewidths=1.0)
    if coronary_calcium[k].any():
        ax.contour(x_coords, y_coords, coronary_calcium[k].astype(float), levels=[0.5], colors=["#ffe040"], linewidths=1.4)
    ax.set_xlim(-155, 145)
    ax.set_ylim(170, -130)
    ax.set_title(f"z={actual_z:.0f} mm", color="white", fontsize=9)
    ax.axis("off")
fig.suptitle(
    "Heart/coronary CT approximation | red=heart envelope | yellow=calcium candidates",
    color="white",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT / "heart_coronary_validation.png", dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)

voxel_ml = np.prod(spacing) / 1000
report = f"""心脏与冠状动脉钙化候选三维模型

来源：无增强、非心电门控全身CT
心脏区域近似体积：{heart.sum() * voxel_ml:.1f} mL
冠脉区域内高密度候选体积：{coronary_calcium.sum() * voxel_ml:.2f} mL

限制：
- 心脏模型是心脏/心包区域近似外形，不代表心腔或心肌精确边界。
- 本数据在排除脊柱等骨结构后，仅剩极少高密度候选，无法可靠重建冠状动脉。
- 黄色模型仅为冠脉常见区域内残留高密度候选，可能是噪声、瓣膜或运动伪影。
- 不能显示完整冠状动脉管腔，不能判断狭窄程度、软斑块或进行正式钙化积分。
- 若要可靠冠脉建模，需要心电门控冠状动脉CTA；若评估钙化积分，需要专用心电门控平扫。
"""
(OUT / "heart_coronary_README.txt").write_text(report, encoding="utf-8")
print(report)
