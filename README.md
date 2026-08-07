# PET/CT 3D Interactive Medical Visualization Suite

A high-performance 3D surface rendering and organ segmentation viewer for PET/CT medical DICOM datasets, built with Python and VTK.

---

## 📸 3D Rendered Visualizations & Screenshot Gallery

### 1. Full Body Surface & Skeletal Overview (Anterior View)
![Full Frontal Overview](docs/images/01_full_overview_front.png)

### 2. 3D Oblique View (Multi-Organ Overlay)
![Oblique View](docs/images/02_oblique_3d_view.png)

### 3. Skeletal & Cardiovascular Reconstruction
![Bone and Cardiovascular](docs/images/03_bone_and_cardiovascular.png)

### 4. High-Resolution Thyroid & Cervical Spine Segmentation Detail
![Thyroid and Neck Detail](docs/images/04_thyroid_neck_detail.png)

---

## ✨ Features

- **Interactive 3D VTK Renderer**: Real-time trackball camera rotation, pan, zoom, and smooth lighting shaders.
- **Multi-Organ Layer Toggling**: Instantly show or hide specific anatomical structures using keyboard shortcuts.
- **Anatomical Segmentations Supported**:
  - 🦴 Skeletal System (Bone Surface)
  - 👤 Full Body Contour Surface
  - 🦋 Thyroid Gland (CT-derived approximation & PET candidates)
  - ❤️ Cardiac & Cardiovascular Structures (Heart, Coronary Candidates)
  - 🩺 Cervical Spine Detailed CT Mesh

---

## 🎮 Interactive Controls

### Mouse Navigation
- **Left Mouse Drag**: Rotate 3D Model
- **Right Mouse Drag**: Pan Camera
- **Mouse Wheel**: Zoom In / Zoom Out

### Keyboard Layer Controls
| Key | Layer Description |
| :---: | :--- |
| `1` | Toggle Full Body Surface Mesh |
| `2` | Toggle Bone / Skeletal Surface Mesh |
| `3` | Toggle PET/CT Thyroid Candidate Segmentation |
| `4` | Toggle CT-derived Thyroid Mesh |
| `5` | Toggle Cardiac / Heart Model |
| `6` | Toggle Coronary Calcium Candidate Mesh |
| `7` | Toggle Cervical Spine Mesh |
| `R` | Reset Camera View |
| `Q` / `ESC` | Exit Application |

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- VTK (`pip install vtk`)
- PyDICOM (`pip install pydicom`)
- NumPy & Matplotlib

### 2. Running on Windows
Simply double-click [`start_3d_viewer.bat`](start_3d_viewer.bat) or run from terminal:
```bash
py interactive_3d_viewer.py
```

### 3. Generate High-Res Screenshots
To automatically capture off-screen 3D renderings:
```bash
py generate_github_screenshots.py
```

---

## 📁 Repository Structure

```text
petct-3d-viewer/
├── docs/
│   └── images/                     # 3D Screenshots for GitHub documentation
│       ├── 01_full_overview_front.png
│       ├── 02_oblique_3d_view.png
│       ├── 03_bone_and_cardiovascular.png
│       └── 04_thyroid_neck_detail.png
├── interactive_3d_viewer.py        # Primary interactive VTK 3D viewer
├── generate_github_screenshots.py  # Off-screen screenshot generator
├── analyze_thyroid_petct.py       # PET/CT quantitative thyroid analysis
├── segment_thyroid_from_ct.py     # CT thyroid segmentation pipeline
├── segment_heart_coronary.py      # Cardiac segmentation pipeline
├── segment_cervical_spine.py      # Cervical spine segmentation pipeline
├── start_3d_viewer.bat            # Windows 1-click launcher
└── .gitignore                      # Excludes heavy raw DICOM data
```

---

## ⚖️ License & Medical Disclaimer
*Note: CT and PET derived organ models in this repository are for 3D visualization research and educational purposes.*
