"""
Generate multi-angle 3D renderings and create a rotating animation for 3D visualization.
"""
import pydicom
import numpy as np
import os
import glob
import vtk

base = r'D:\边绍康华西检查\打包 (2) (1)\打包_data'
output_dir = os.path.join(base, 'images')

def load_volume(data_folder):
    """Load DICOM series and return VTK image data."""
    dcm_files = sorted(glob.glob(os.path.join(data_folder, '*.dcm')))
    slices = []
    for f in dcm_files:
        try:
            slices.append(pydicom.dcmread(f, force=True))
        except:
            pass
    if not slices:
        return None, None
    
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except:
        pass
    
    first = slices[0]
    rows, cols = int(first.Rows), int(first.Columns)
    num = len(slices)
    
    try:
        ps = [float(v) for v in first.PixelSpacing]
        sx, sy = ps[1], ps[0]
    except:
        sx = sy = 1.0
    
    if num > 1:
        try:
            zpos = [float(s.ImagePositionPatient[2]) for s in slices]
            sz = abs(zpos[-1] - zpos[0]) / (num - 1)
        except:
            sz = 1.0
    else:
        sz = 1.0
    
    arr = np.zeros((num, rows, cols), dtype=np.float32)
    for i, s in enumerate(slices):
        try:
            a = s.pixel_array.astype(np.float32)
            slope = float(getattr(s, 'RescaleSlope', 1))
            intercept = float(getattr(s, 'RescaleIntercept', 0))
            arr[i] = a * slope + intercept
        except:
            pass
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(cols, rows, num)
    vtk_img.SetSpacing(sx, sy, sz)
    vtk_img.AllocateScalars(vtk.VTK_FLOAT, 1)
    flat = arr.flatten(order='F')
    va = vtk.vtkFloatArray()
    va.SetNumberOfValues(len(flat))
    for i in range(len(flat)):
        va.SetValue(i, flat[i])
    vtk_img.GetPointData().SetScalars(va)
    
    return vtk_img, (cols, rows, num, sx, sy, sz)


def render_angle(ct_image, pet_image=None, angle=0, elevation=10):
    """Render volume from a specific angle."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.02, 0.02, 0.05)
    
    # CT
    if ct_image:
        ct_prop = vtk.vtkVolumeProperty()
        ct_prop.ShadeOn()
        ct_prop.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
        
        ct_color = vtk.vtkColorTransferFunction()
        ct_color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        ct_color.AddRGBPoint(-400, 0.2, 0.2, 0.2)
        ct_color.AddRGBPoint(0, 0.3, 0.3, 0.3)
        ct_color.AddRGBPoint(300, 0.5, 0.5, 0.5)
        ct_color.AddRGBPoint(1000, 0.9, 0.9, 0.9)
        ct_color.AddRGBPoint(2000, 1.0, 1.0, 1.0)
        
        ct_opacity = vtk.vtkPiecewiseFunction()
        ct_opacity.AddPoint(-1000, 0.0)
        ct_opacity.AddPoint(-500, 0.0)
        ct_opacity.AddPoint(-200, 0.01)
        ct_opacity.AddPoint(100, 0.05)
        ct_opacity.AddPoint(300, 0.2)
        ct_opacity.AddPoint(600, 0.5)
        ct_opacity.AddPoint(1000, 0.85)
        ct_opacity.AddPoint(2000, 1.0)
        
        ct_prop.SetColor(ct_color)
        ct_prop.SetScalarOpacity(ct_opacity)
        
        ct_mapper = vtk.vtkSmartVolumeMapper()
        ct_mapper.SetInputData(ct_image)
        ct_mapper.SetRequestedRenderModeToDefault()
        
        ct_vol = vtk.vtkVolume()
        ct_vol.SetMapper(ct_mapper)
        ct_vol.SetProperty(ct_prop)
        renderer.AddVolume(ct_vol)
    
    # PET
    if pet_image:
        pet_prop = vtk.vtkVolumeProperty()
        pet_prop.ShadeOff()
        pet_prop.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
        
        pet_color = vtk.vtkColorTransferFunction()
        pet_data = pet_image.GetPointData().GetScalars()
        p_min, p_max = pet_data.GetRange(0)
        p_range = max(p_max - p_min, 0.001)
        p_50 = p_min + p_range * 0.5
        p_75 = p_min + p_range * 0.75
        p_90 = p_min + p_range * 0.9
        
        pet_color.AddRGBPoint(p_min, 0.0, 0.0, 0.0)
        pet_color.AddRGBPoint(p_min + p_range*0.2, 0.0, 0.0, 0.0)
        pet_color.AddRGBPoint(p_50, 0.0, 0.0, 0.5)
        pet_color.AddRGBPoint(p_75, 1.0, 0.0, 0.0)
        pet_color.AddRGBPoint(p_90, 1.0, 1.0, 0.0)
        pet_color.AddRGBPoint(p_max, 1.0, 1.0, 1.0)
        
        pet_opacity = vtk.vtkPiecewiseFunction()
        pet_opacity.AddPoint(p_min, 0.0)
        pet_opacity.AddPoint(p_min + p_range*0.2, 0.0)
        pet_opacity.AddPoint(p_50, 0.0)
        pet_opacity.AddPoint(p_75, 0.35)
        pet_opacity.AddPoint(p_90, 0.7)
        pet_opacity.AddPoint(p_max, 1.0)
        
        pet_prop.SetColor(pet_color)
        pet_prop.SetScalarOpacity(pet_opacity)
        
        # Resample PET to CT space if both exist
        if ct_image:
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(pet_image)
            reslice.SetOutputOrigin(ct_image.GetOrigin())
            reslice.SetOutputSpacing(ct_image.GetSpacing())
            reslice.SetOutputExtent(ct_image.GetExtent())
            reslice.Update()
            pet_input = reslice.GetOutput()
        else:
            pet_input = pet_image
        
        pet_mapper = vtk.vtkSmartVolumeMapper()
        pet_mapper.SetInputData(pet_input)
        pet_mapper.SetRequestedRenderModeToDefault()
        
        pet_vol = vtk.vtkVolume()
        pet_vol.SetMapper(pet_mapper)
        pet_vol.SetProperty(pet_prop)
        renderer.AddVolume(pet_vol)
    
    # Calculate max dimension for camera distance
    if ct_image:
        dims = ct_image.GetDimensions()
        spacing = ct_image.GetSpacing()
    elif pet_image:
        dims = pet_image.GetDimensions()
        spacing = pet_image.GetSpacing()
    else:
        return None
    
    max_dim = max(dims[0]*spacing[0], dims[1]*spacing[1], dims[2]*spacing[2]) * 0.5
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(1000, 800)
    render_window.AddRenderer(renderer)
    
    camera = renderer.GetActiveCamera()
    rad = np.radians(angle)
    elev = np.radians(elevation)
    dist = max_dim * 3.5
    cx = dist * np.cos(elev) * np.sin(rad)
    cy = -dist * np.cos(elev) * np.cos(rad)
    cz = dist * np.sin(elev)
    
    camera.SetPosition(cx, cy, cz)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    
    render_window.Render()
    
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.Update()
    
    return w2if.GetOutput()


# Load data
print("Loading CT WB...")
ct_img, ct_params = load_volume(os.path.join(base, 'Data8'))
print(f"  CT: {ct_params[0]}x{ct_params[1]}x{ct_params[2]}, spacing={ct_params[3]:.2f}x{ct_params[4]:.2f}x{ct_params[5]:.2f}")

print("Loading PET WB...")
pet_img, pet_params = load_volume(os.path.join(base, 'Data1'))
print(f"  PET: {pet_params[0]}x{pet_params[1]}x{pet_params[2]}, spacing={pet_params[3]:.2f}x{pet_params[4]:.2f}x{pet_params[5]:.2f}")

# Generate multi-angle fusion renderings
print("\nGenerating multi-angle PET/CT fusion renderings...")
angles = list(range(0, 360, 30))  # 12 angles
fusion_images = []

for angle in angles:
    print(f"  Angle {angle}°...", end=" ")
    img = render_angle(ct_img, pet_img, angle=angle, elevation=15)
    if img:
        writer = vtk.vtkPNGWriter()
        path = os.path.join(output_dir, f'fusion_angle_{angle:03d}.png')
        writer.SetFileName(path)
        writer.SetInputData(img)
        writer.Write()
        fusion_images.append(path)
        print("OK")

# Also generate CT-only multi-angle
print("\nGenerating multi-angle CT renderings...")
ct_angles = []
for angle in angles:
    print(f"  Angle {angle}°...", end=" ")
    img = render_angle(ct_img, None, angle=angle, elevation=15)
    if img:
        writer = vtk.vtkPNGWriter()
        path = os.path.join(output_dir, f'ct_angle_{angle:03d}.png')
        writer.SetFileName(path)
        writer.SetInputData(img)
        writer.Write()
        ct_angles.append(path)
        print("OK")

print(f"\nDone! Generated {len(fusion_images)} fusion + {len(ct_angles)} CT multi-angle views.")
