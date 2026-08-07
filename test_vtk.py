import vtk
from vtkmodules.vtkRenderingCore import vtkRenderWindow

rw = vtkRenderWindow()
rw.SetOffScreenRendering(1)
print('VTK offscreen rendering available')
print(f'VTK Version: {vtk.vtkVersion.GetVTKVersion()}')