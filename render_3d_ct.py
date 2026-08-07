import pydicom
import numpy as np
import os, glob
from vtkmodules.all import (
    vtkImageImport, vtkSmartVolumeMapper, vtkVolume,
    vtkVolumeProperty, vtkColorTransferFunction,
    vtkPiecewiseFunction, vtkRenderer, vtkRenderWindow,
    vtkRenderWindowInteractor, vtkWindowToImageFilter,
    vtkPNGWriter, vtkImageReslice, vtkImageCast,
    vtkCamera, vtkFixedPointVolumeRayCastMapper
)

data_root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(data_root, "images")
os.makedirs(output_dir, exist_ok=True)

def load_dicom_series(data_folder, sort_by='z'):
    folder = os.path.join(data_root, data_folder)
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        return None, None
    slices = []
    for f in files:
        ds = pydicom.dcmread(f, force=True)
        slices.append(ds)
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    img = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    slope = float(getattr(slices[0], 'RescaleSlope', 1))
    intercept = float(getattr(slices[0], 'RescaleIntercept', 0))
    img = img * slope + intercept

    info = {
        'modality': getattr(slices[0], 'Modality', 'Unknown'),
        'series_desc': getattr(slices[0], 'SeriesDescription', 'Unknown'),
        'shape': img.shape,
        'spacing': [
            float(slices[0].PixelSpacing[0]),
            float(slices[0].PixelSpacing[1]),
            abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
        ] if len(slices) > 1 else [1,1,1],
        'rows': int(slices[0].Rows),
        'columns': int(slices[0].Columns),
        'num_slices': len(slices)
    }
    return img, info

def numpy_to_vtk_image(volume, spacing, origin=(0,0,0)):
    imp = vtkImageImport()
    data_bytes = np.ascontiguousarray(volume[::-1].tobytes())
    imp.CopyImportVoidPointer(data_bytes, len(data_bytes))
    imp.SetDataScalarTypeToFloat()
    imp.SetNumberOfScalarComponents(1)
    dims = volume.shape
    ext = (0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
    imp.SetWholeExtent(*ext)
    imp.SetDataExtent(*ext)
    imp.SetDataSpacing(*spacing)
    imp.SetDataOrigin(*origin)
    imp.Update()
    return imp.GetOutput()

def make_mapper(image_data, blend_mode='composite'):
    """创建基于CPU的固定点体绘制映射器"""
    mapper = vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputData(image_data)
    if blend_mode == 'mip':
        mapper.SetBlendModeToMaximumIntensity()
    else:
        mapper.SetBlendModeToComposite()
    return mapper

def create_ct_volume_rendering(volume_data, info, output_path):
    """CT体绘制 - 骨骼高亮"""
    spacing = info['spacing']
    data = numpy_to_vtk_image(volume_data, spacing)
    mapper = make_mapper(data, 'composite')

    ctf = vtkColorTransferFunction()
    ctf.AddRGBPoint(-200, 0.0, 0.0, 0.0)
    ctf.AddRGBPoint(0, 0.5, 0.5, 0.5)
    ctf.AddRGBPoint(300, 0.9, 0.85, 0.7)
    ctf.AddRGBPoint(800, 1.0, 0.95, 0.8)
    ctf.AddRGBPoint(1200, 1.0, 1.0, 1.0)

    otf = vtkPiecewiseFunction()
    otf.AddPoint(-200, 0.0)
    otf.AddPoint(0, 0.05)
    otf.AddPoint(150, 0.1)
    otf.AddPoint(300, 0.3)
    otf.AddPoint(600, 0.6)
    otf.AddPoint(1000, 0.85)
    otf.AddPoint(1500, 0.95)

    prop = vtkVolumeProperty()
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer = vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.02, 0.02, 0.04)

    render_window = vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(800, 600)
    render_window.AddRenderer(renderer)

    camera = renderer.GetActiveCamera()
    bounds = data.GetBounds()
    max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    dist = max_dim * 1.8
    cx = bounds[0] + (bounds[1]-bounds[0])/2
    cy = bounds[2] + (bounds[3]-bounds[2])/2
    cz = bounds[4] + (bounds[5]-bounds[4])/2
    camera.SetPosition(cx + dist*0.3, cy - dist*0.3, cz + dist*0.3)
    camera.SetFocalPoint(cx, cy, cz)
    camera.SetViewUp(0, 0, 1)

    render_window.Render()

    w2if = vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.Update()

    writer = vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
    return True

def create_mip_rendering(volume_data, info, output_path, modality="CT"):
    """MIP最大密度投影"""
    spacing = info['spacing']
    data = numpy_to_vtk_image(volume_data, spacing)
    mapper = make_mapper(data, 'mip')

    ctf = vtkColorTransferFunction()
    otf = vtkPiecewiseFunction()

    if modality == "PET":
        ctf.AddRGBPoint(0, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(volume_data.max()*0.1, 0.5, 0.0, 0.0)
        ctf.AddRGBPoint(volume_data.max()*0.3, 1.0, 0.5, 0.0)
        ctf.AddRGBPoint(volume_data.max()*0.5, 1.0, 1.0, 0.0)
        ctf.AddRGBPoint(volume_data.max()*0.7, 1.0, 1.0, 0.5)
        ctf.AddRGBPoint(volume_data.max()*0.9, 1.0, 1.0, 1.0)
        otf.AddPoint(0, 0.0)
        otf.AddPoint(volume_data.max()*0.05, 1.0)
    else:
        ctf.AddRGBPoint(-200, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(0, 0.3, 0.3, 0.3)
        ctf.AddRGBPoint(300, 0.7, 0.7, 0.7)
        ctf.AddRGBPoint(800, 0.9, 0.9, 0.9)
        ctf.AddRGBPoint(1200, 1.0, 1.0, 1.0)
        otf.AddPoint(0, 0.0)
        otf.AddPoint(100, 1.0)

    prop = vtkVolumeProperty()
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)
    prop.SetInterpolationTypeToLinear()

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer = vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.02, 0.02, 0.04)

    render_window = vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(800, 600)
    render_window.AddRenderer(renderer)

    camera = renderer.GetActiveCamera()
    bounds = data.GetBounds()
    max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    dist = max_dim * 1.8
    cx = bounds[0] + (bounds[1]-bounds[0])/2
    cy = bounds[2] + (bounds[3]-bounds[2])/2
    cz = bounds[4] + (bounds[5]-bounds[4])/2
    camera.SetPosition(cx + dist*0.3, cy - dist*0.3, cz + dist*0.3)
    camera.SetFocalPoint(cx, cy, cz)
    camera.SetViewUp(0, 0, 1)

    render_window.Render()

    w2if = vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.Update()

    writer = vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
    return True

def create_pet_ct_fusion(pet_data, pet_info, ct_data, ct_info, output_path):
    """PET/CT融合"""
    ct_spacing = ct_info['spacing']
    ct_image = numpy_to_vtk_image(ct_data, ct_spacing)

    pet_spacing = pet_info['spacing']
    pet_vtk = numpy_to_vtk_image(pet_data, pet_spacing)

    # Resample PET to CT space
    reslice = vtkImageReslice()
    reslice.SetInputData(pet_vtk)
    reslice.SetOutputSpacing(ct_spacing[0], ct_spacing[1], ct_spacing[2])
    reslice.SetInterpolationModeToLinear()
    reslice.Update()
    pet_resampled = reslice.GetOutput()

    # CT rendering (CPU-based)
    mapper_ct = vtkFixedPointVolumeRayCastMapper()
    mapper_ct.SetInputData(ct_image)
    mapper_ct.SetBlendModeToComposite()

    ctf_ct = vtkColorTransferFunction()
    ctf_ct.AddRGBPoint(-200, 0.0, 0.0, 0.0)
    ctf_ct.AddRGBPoint(0, 0.2, 0.2, 0.2)
    ctf_ct.AddRGBPoint(300, 0.5, 0.5, 0.5)
    ctf_ct.AddRGBPoint(800, 0.7, 0.7, 0.7)
    ctf_ct.AddRGBPoint(1200, 1.0, 1.0, 1.0)

    otf_ct = vtkPiecewiseFunction()
    otf_ct.AddPoint(-200, 0.0)
    otf_ct.AddPoint(0, 0.05)
    otf_ct.AddPoint(150, 0.08)
    otf_ct.AddPoint(300, 0.15)
    otf_ct.AddPoint(600, 0.4)
    otf_ct.AddPoint(1000, 0.7)

    prop_ct = vtkVolumeProperty()
    prop_ct.SetColor(ctf_ct)
    prop_ct.SetScalarOpacity(otf_ct)
    prop_ct.ShadeOn()
    prop_ct.SetInterpolationTypeToLinear()

    vol_ct = vtkVolume()
    vol_ct.SetMapper(mapper_ct)
    vol_ct.SetProperty(prop_ct)

    # PET rendering (CPU-based)
    mapper_pet = vtkFixedPointVolumeRayCastMapper()
    mapper_pet.SetInputData(pet_resampled)
    mapper_pet.SetBlendModeToComposite()

    pet_max = float(pet_data.max())

    ctf_pet = vtkColorTransferFunction()
    ctf_pet.AddRGBPoint(0, 0.0, 0.0, 0.0)
    ctf_pet.AddRGBPoint(pet_max*0.15, 0.0, 0.0, 0.5)
    ctf_pet.AddRGBPoint(pet_max*0.3, 0.0, 0.5, 1.0)
    ctf_pet.AddRGBPoint(pet_max*0.45, 0.0, 1.0, 0.5)
    ctf_pet.AddRGBPoint(pet_max*0.6, 0.5, 1.0, 0.0)
    ctf_pet.AddRGBPoint(pet_max*0.75, 1.0, 0.8, 0.0)
    ctf_pet.AddRGBPoint(pet_max*0.9, 1.0, 0.2, 0.0)

    otf_pet = vtkPiecewiseFunction()
    otf_pet.AddPoint(0, 0.0)
    otf_pet.AddPoint(pet_max*0.1, 0.0)
    otf_pet.AddPoint(pet_max*0.25, 0.3)
    otf_pet.AddPoint(pet_max*0.5, 0.7)
    otf_pet.AddPoint(pet_max*0.8, 1.0)

    prop_pet = vtkVolumeProperty()
    prop_pet.SetColor(ctf_pet)
    prop_pet.SetScalarOpacity(otf_pet)
    prop_pet.SetInterpolationTypeToLinear()

    vol_pet = vtkVolume()
    vol_pet.SetMapper(mapper_pet)
    vol_pet.SetProperty(prop_pet)

    # Render
    renderer = vtkRenderer()
    renderer.AddVolume(vol_ct)
    renderer.AddVolume(vol_pet)
    renderer.SetBackground(0.02, 0.02, 0.04)

    render_window = vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(800, 600)
    render_window.AddRenderer(renderer)

    camera = renderer.GetActiveCamera()
    bounds = ct_image.GetBounds()
    max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    dist = max_dim * 1.8
    cx = bounds[0] + (bounds[1]-bounds[0])/2
    cy = bounds[2] + (bounds[3]-bounds[2])/2
    cz = bounds[4] + (bounds[5]-bounds[4])/2
    camera.SetPosition(cx + dist*0.3, cy - dist*0.3, cz + dist*0.3)
    camera.SetFocalPoint(cx, cy, cz)
    camera.SetViewUp(0, 0, 1)

    render_window.Render()

    w2if = vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.Update()

    writer = vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
    return True


# ===== Main =====
print("Loading CT WB data (Data8)...")
ct_data, ct_info = load_dicom_series("Data8")
if ct_data is None:
    print("ERROR: CT data not found in Data8")
    exit(1)
print(f"  CT shape: {ct_data.shape}, spacing: {ct_info['spacing']}")

print("Loading PET WB data (Data1)...")
pet_data, pet_info = load_dicom_series("Data1")
if pet_data is None:
    print("ERROR: PET data not found in Data1")
    exit(1)
print(f"  PET shape: {pet_data.shape}, spacing: {pet_info['spacing']}")

print("Clamping PET to 99.5% percentile...")
p99_5 = np.percentile(pet_data, 99.5)
pet_data = np.clip(pet_data, 0, p99_5)
print(f"  PET clamped to {p99_5:.1f}")

print("Generating CT volume rendering...")
create_ct_volume_rendering(ct_data, ct_info, os.path.join(output_dir, "ct_volume.png"))
print("  -> ct_volume.png OK")

print("Generating CT MIP...")
create_mip_rendering(ct_data, ct_info, os.path.join(output_dir, "ct_mip.png"), "CT")
print("  -> ct_mip.png OK")

print("Generating PET MIP...")
create_mip_rendering(pet_data, pet_info, os.path.join(output_dir, "pet_mip.png"), "PET")
print("  -> pet_mip.png OK")

print("Generating PET/CT fusion...")
create_pet_ct_fusion(pet_data, pet_info, ct_data, ct_info, os.path.join(output_dir, "pet_ct_fusion.png"))
print("  -> pet_ct_fusion.png OK")

print("\nAll 4 main renderings generated successfully in images/")
