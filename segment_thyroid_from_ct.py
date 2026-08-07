"""Approximate patient-specific thyroid segmentation from non-contrast CT.

The result is intended for visualization only. Non-contrast whole-body CT
does not provide enough soft-tissue contrast for clinical contouring.
"""
from collections import deque
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


def xyz_to_zyx(x, y, z):
    return (
        int(round((z - origin[2]) / spacing[2])),
        int(round((y - origin[1]) / spacing[1])),
        int(round((x - origin[0]) / spacing[0])),
    )


def region_grow_2d(slice_hu, allowed, seeds):
    out = np.zeros_like(allowed, dtype=bool)
    q = deque()
    for y, x in seeds:
        if 0 <= y < allowed.shape[0] and 0 <= x < allowed.shape[1] and allowed[y, x]:
            out[y, x] = True
            q.append((y, x))
    while q:
        y, x = q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < allowed.shape[0] and 0 <= xx < allowed.shape[1]:
                if allowed[yy, xx] and not out[yy, xx]:
                    out[yy, xx] = True
                    q.append((yy, xx))
    return out


def ellipse_mask(shape, cx, cy, rx, ry):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def binary_close_2d(mask, iterations=2):
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1)
        out = np.logical_or.reduce(
            [padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
             for dy in range(3) for dx in range(3)]
        )
    for _ in range(iterations):
        padded = np.pad(out, 1, constant_values=True)
        out = np.logical_and.reduce(
            [padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
             for dy in range(3) for dx in range(3)]
        )
    return out


def vtk_image(array):
    image = vtk.vtkImageData()
    image.SetDimensions(array.shape[2], array.shape[1], array.shape[0])
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    scalars = numpy_to_vtk(
        np.ascontiguousarray(array.astype(np.uint8)).ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    image.GetPointData().SetScalars(scalars)
    return image


ct, origin, spacing = load_ct()
mask = np.zeros_like(ct, dtype=bool)

# Slice-by-slice constrained region growing. The anatomical envelopes keep
# muscle, vessels, airway and esophagus from joining the thyroid mask.
for z_mm in np.arange(-360, -327, spacing[2]):
    k = xyz_to_zyx(0, 0, z_mm)[0]
    if not 0 <= k < ct.shape[0]:
        continue
    # Lobes become slightly wider and more posterior toward the lower pole.
    t = np.clip((-342 - z_mm) / 15, -1, 1)
    # On these images the trachea is centered near x=-5, y=-22 mm. Thyroid
    # tissue lies immediately lateral and slightly posterior to it.
    cy_mm = -20 + 1.5 * t
    rx_mm = 13.0 - 2.0 * abs(t)
    ry_mm = 16.0 - 2.0 * abs(t)
    allowed_hu = (ct[k] >= 15) & (ct[k] <= 145)
    slice_mask = np.zeros(ct.shape[1:], dtype=bool)
    for cx_mm in (-22.0, 12.0):
        _, cy, cx = xyz_to_zyx(cx_mm, cy_mm, z_mm)
        env = ellipse_mask(
            ct.shape[1:],
            cx,
            cy,
            rx_mm / spacing[0],
            ry_mm / spacing[1],
        )
        slice_mask |= allowed_hu & env
    # Approximate the thin isthmus only on central slices.
    if -348 <= z_mm <= -336:
        _, cy, cx = xyz_to_zyx(-5, -35, z_mm)
        isthmus = ellipse_mask(ct.shape[1:], cx, cy, 15 / spacing[0], 4 / spacing[1])
        slice_mask |= allowed_hu & isthmus
    mask[k] = binary_close_2d(slice_mask, iterations=2)

# Smooth and keep the resulting surface compact.
img = vtk_image(mask)
smooth = vtk.vtkImageGaussianSmooth()
smooth.SetInputData(img)
smooth.SetStandardDeviations(1.2, 1.2, 0.8)
smooth.SetRadiusFactors(2, 2, 2)
smooth.Update()

contour = vtk.vtkFlyingEdges3D()
contour.SetInputConnection(smooth.GetOutputPort())
contour.SetValue(0, 0.38)

poly_smooth = vtk.vtkWindowedSincPolyDataFilter()
poly_smooth.SetInputConnection(contour.GetOutputPort())
poly_smooth.SetNumberOfIterations(20)
poly_smooth.SetPassBand(0.08)
poly_smooth.BoundarySmoothingOff()
poly_smooth.Update()

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(str(OUT / "thyroid_ct_approx.vtp"))
writer.SetInputConnection(poly_smooth.GetOutputPort())
writer.Write()

stl = vtk.vtkSTLWriter()
stl.SetFileName(str(OUT / "thyroid_ct_approx.stl"))
stl.SetInputConnection(poly_smooth.GetOutputPort())
stl.Write()

np.savez_compressed(
    OUT / "thyroid_ct_approx_mask.npz",
    mask=mask,
    origin=origin,
    spacing=spacing,
)

# Standalone 3D preview.
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(poly_smooth.GetOutputPort())
mapper.ScalarVisibilityOff()
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.95, 0.18, 0.42)
actor.GetProperty().SetSpecular(0.3)
actor.GetProperty().SetSpecularPower(25)
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.015, 0.02, 0.035)
renderer.AddActor(actor)
window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetSize(1000, 800)
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
png.SetFileName(str(OUT / "thyroid_ct_approx_3d.png"))
png.SetInputConnection(capture.GetOutputPort())
png.Write()

# Validation montage with red CT-derived contour.
fig, axes = plt.subplots(3, 4, figsize=(12, 10), facecolor="#05070d")
z_targets = np.linspace(-360, -327, 12)
extent = [
    origin[0],
    origin[0] + spacing[0] * ct.shape[2],
    origin[1] + spacing[1] * ct.shape[1],
    origin[1],
]
x_coords = origin[0] + np.arange(ct.shape[2]) * spacing[0]
y_coords = origin[1] + np.arange(ct.shape[1]) * spacing[1]
for ax, z_mm in zip(axes.flat, z_targets):
    k = xyz_to_zyx(0, 0, z_mm)[0]
    actual_z = origin[2] + k * spacing[2]
    ax.imshow(ct[k], cmap="gray", vmin=-120, vmax=260, extent=extent)
    ax.contour(
        x_coords,
        y_coords,
        mask[k].astype(float),
        levels=[0.5],
        colors=["#ff315f"],
        linewidths=1.4,
    )
    ax.set_xlim(-75, 75)
    ax.set_ylim(65, -80)
    ax.set_title(f"z={actual_z:.0f} mm", color="white", fontsize=9)
    ax.axis("off")
fig.suptitle(
    "Approximate CT-derived thyroid contour (red) - visualization only",
    color="white",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(OUT / "thyroid_ct_approx_validation.png", dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)

mass = vtk.vtkMassProperties()
tri = vtk.vtkTriangleFilter()
tri.SetInputConnection(poly_smooth.GetOutputPort())
mass.SetInputConnection(tri.GetOutputPort())
mass.Update()
volume_ml = mass.GetVolume() / 1000

report = f"""CT辅助甲状腺近似分割

来源：Data8 全身无增强CT，软组织重建核
分辨率：{spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm
近似模型体积：{volume_ml:.1f} mL

文件：
- thyroid_ct_approx.vtp：ParaView / 3D Slicer 可打开
- thyroid_ct_approx.stl：通用三维网格
- thyroid_ct_approx_validation.png：逐层红色轮廓验证图
- thyroid_ct_approx_3d.png：单独三维预览
- thyroid_ct_approx_mask.npz：分割掩膜

重要限制：
- 这是从无增强、3 mm层厚CT生成的研究级近似分割，不是医生逐层勾画结果。
- 甲状腺与周围肌肉、血管对比有限，模型边界和体积可能存在明显误差。
- 模型不能用于判断结节性质、手术规划或放疗计划。
"""
(OUT / "thyroid_ct_approx_README.txt").write_text(report, encoding="utf-8")
print(report)
