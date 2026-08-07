"""Create a cautious thyroid-region PET/CT review sheet.

This is a localization aid, not an automated diagnosis.
"""
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pydicom


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "organ_3d"


def load(folder):
    dss = [pydicom.dcmread(f, force=True) for f in glob.glob(str(ROOT / folder / "*.dcm"))]
    dss.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    first = dss[0]
    arr = np.stack([x.pixel_array for x in dss]).astype(np.float32)
    arr = arr * float(getattr(first, "RescaleSlope", 1)) + float(getattr(first, "RescaleIntercept", 0))
    origin = np.array([float(v) for v in first.ImagePositionPatient])
    spacing = np.array([
        float(first.PixelSpacing[1]),
        float(first.PixelSpacing[0]),
        np.median(np.diff([float(x.ImagePositionPatient[2]) for x in dss])),
    ])
    return arr, origin, spacing, first


def index_from_xyz(xyz, origin, spacing):
    x, y, z = xyz
    return np.array([
        int(round((z - origin[2]) / spacing[2])),
        int(round((y - origin[1]) / spacing[1])),
        int(round((x - origin[0]) / spacing[0])),
    ])


def pet_on_ct_slice(ct_shape, z, pet, pet_origin, pet_spacing):
    y = ct_origin[1] + np.arange(ct_shape[0]) * ct_spacing[1]
    x = ct_origin[0] + np.arange(ct_shape[1]) * ct_spacing[0]
    pz = int(round((z - pet_origin[2]) / pet_spacing[2]))
    py = np.rint((y - pet_origin[1]) / pet_spacing[1]).astype(int)
    px = np.rint((x - pet_origin[0]) / pet_spacing[0]).astype(int)
    py = np.clip(py, 0, pet.shape[1] - 1)
    px = np.clip(px, 0, pet.shape[2] - 1)
    return pet[pz][np.ix_(py, px)]


ct, ct_origin, ct_spacing, ct_ds = load("Data8")
pet_bqml, pet_origin, pet_spacing, pet_ds = load("Data1")

weight_g = float(pet_ds.PatientWeight) * 1000
info = pet_ds.RadiopharmaceuticalInformationSequence[0]
dose_bq = float(info.RadionuclideTotalDose)
pet_suv = pet_bqml * weight_g / dose_bq

# Conservative lobe review boxes around the lower anterior neck. The central
# airway and the more superior salivary glands are excluded.
box = {
    "x": (-38.0, 38.0),
    "y": (-70.0, -15.0),
    "z": (-345.0, -315.0),
}

xv = pet_origin[0] + np.arange(pet_suv.shape[2]) * pet_spacing[0]
yv = pet_origin[1] + np.arange(pet_suv.shape[1]) * pet_spacing[1]
zv = pet_origin[2] + np.arange(pet_suv.shape[0]) * pet_spacing[2]
xi = np.where((xv >= box["x"][0]) & (xv <= box["x"][1]))[0]
yi = np.where((yv >= box["y"][0]) & (yv <= box["y"][1]))[0]
zi = np.where((zv >= box["z"][0]) & (zv <= box["z"][1]))[0]
roi = pet_suv[np.ix_(zi, yi, xi)].copy()
# Exclude the central airway from the maximum search.
roi[:, :, np.abs(xv[xi]) < 8] = -1
peak_local = np.unravel_index(np.argmax(roi), roi.shape)
peak_ijk = (zi[peak_local[0]], yi[peak_local[1]], xi[peak_local[2]])
peak_xyz = (
    float(xv[peak_ijk[2]]),
    float(yv[peak_ijk[1]]),
    float(zv[peak_ijk[0]]),
)
peak_suv = float(pet_suv[peak_ijk])

xpos = roi[:, :, xv[xi] >= 8]
xneg = roi[:, :, xv[xi] <= -8]
xpos_max = float(xpos.max())
xneg_max = float(xneg.max())

z_targets = np.linspace(-345, -315, 10)
fig, axes = plt.subplots(2, 5, figsize=(16, 7.2), facecolor="#05070d")
extent = [
    ct_origin[0],
    ct_origin[0] + ct_spacing[0] * ct.shape[2],
    ct_origin[1] + ct_spacing[1] * ct.shape[1],
    ct_origin[1],
]
for ax, target_z in zip(axes.flat, z_targets):
    ci = int(round((target_z - ct_origin[2]) / ct_spacing[2]))
    actual_z = ct_origin[2] + ci * ct_spacing[2]
    suv_slice = pet_on_ct_slice(ct.shape[1:], actual_z, pet_suv, pet_origin, pet_spacing)
    ax.imshow(ct[ci], cmap="gray", vmin=-120, vmax=280, extent=extent)
    ax.imshow(
        np.ma.masked_less(suv_slice, 0.5),
        cmap="hot",
        vmin=0.5,
        vmax=max(4.0, peak_suv),
        alpha=0.58,
        extent=extent,
    )
    ax.add_patch(
        plt.Rectangle(
            (box["x"][0], box["y"][0]),
            box["x"][1] - box["x"][0],
            box["y"][1] - box["y"][0],
            fill=False,
            edgecolor="#40d9ff",
            linewidth=1.2,
        )
    )
    if abs(actual_z - peak_xyz[2]) <= ct_spacing[2] * 1.5:
        ax.plot(peak_xyz[0], peak_xyz[1], marker="+", color="#00ffff", markersize=14, mew=2)
    ax.set_xlim(-85, 85)
    ax.set_ylim(35, -115)
    ax.set_title(f"z={actual_z:.0f} mm", color="white", fontsize=10)
    ax.axis("off")

fig.suptitle(
    "Thyroid-region PET/CT review | cyan box = review region | + = regional SUVmax\n"
    f"Regional SUVmax {peak_suv:.2f} at x={peak_xyz[0]:.1f}, y={peak_xyz[1]:.1f}, z={peak_xyz[2]:.1f} mm | "
    f"x>=8 max {xpos_max:.2f}, x<=-8 max {xneg_max:.2f}",
    color="white",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT / "thyroid_petct_review.png", dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)

report = f"""甲状腺区域 PET/CT 辅助复核结果

检查类型：Ga-68 PSMA PET/CT
复核区域：x {box['x'][0]} 至 {box['x'][1]} mm；y {box['y'][0]} 至 {box['y'][1]} mm；z {box['z'][0]} 至 {box['z'][1]} mm
区域 SUVmax（近似）：{peak_suv:.2f}
SUVmax 坐标：x={peak_xyz[0]:.1f}, y={peak_xyz[1]:.1f}, z={peak_xyz[2]:.1f} mm
x>=8 mm 一侧 SUVmax：{xpos_max:.2f}
x<=-8 mm 一侧 SUVmax：{xneg_max:.2f}

重要说明：
- 已排除更上方颌下腺和中央气道，但局部最高点仍需影像科医生确认是否确实位于甲状腺。
- Ga-68 PSMA 的甲状腺局灶性摄取既可能是良性，也可能需要进一步检查；仅凭 PET/CT 无法定性。
- 全身 CT 层厚约 3 mm，且不是专门的甲状腺薄层增强 CT，较小结节可能无法可靠识别。
- 是否存在可疑结节，应结合正式 PET/CT 报告、甲状腺超声及必要时穿刺判断。
"""
(OUT / "thyroid_petct_review.txt").write_text(report, encoding="utf-8")
print(report)
