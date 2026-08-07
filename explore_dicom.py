import pydicom, os, glob

base = os.path.dirname(os.path.abspath(__file__))

# Read first DICOM file from Data1
dcm_files = sorted(glob.glob(os.path.join(base, 'Data1', '*.dcm')))
if dcm_files:
    ds = pydicom.dcmread(dcm_files[0], force=True)
    print('=== First DICOM File Info ===')
    print(f'SOP Class UID: {ds.get("SOPClassUID", "N/A")}')
    print(f'Modality: {ds.get("Modality", "N/A")}')
    print(f'Rows: {ds.get("Rows", "N/A")}')
    print(f'Columns: {ds.get("Columns", "N/A")}')
    print(f'Patient Name: {ds.get("PatientName", "N/A")}')
    print(f'Study Description: {ds.get("StudyDescription", "N/A")}')
    print(f'Series Description: {ds.get("SeriesDescription", "N/A")}')
    print(f'Number of frames: {ds.get("NumberOfFrames", "N/A")}')
    print(f'Slice Thickness: {ds.get("SliceThickness", "N/A")}')
    print(f'Spacing Between Slices: {ds.get("SpacingBetweenSlices", "N/A")}')
    print(f'Pixel Spacing: {ds.get("PixelSpacing", "N/A")}')
    print(f'Series Number: {ds.get("SeriesNumber", "N/A")}')
    print(f'Image Position (Patient): {ds.get("ImagePositionPatient", "N/A")}')
    print(f'Image Orientation (Patient): {ds.get("ImageOrientationPatient", "N/A")}')
    print(f'Bits Allocated: {ds.get("BitsAllocated", "N/A")}')
    print(f'Rescale Slope: {ds.get("RescaleSlope", "N/A")}')
    print(f'Rescale Intercept: {ds.get("RescaleIntercept", "N/A")}')
    print(f'Transfer Syntax UID: {ds.file_meta.get("TransferSyntaxUID", "N/A")}')
    
# Now explore each Data folder to find unique series
print('\n=== Exploring Series across Data folders ===')
for d in sorted(glob.glob(os.path.join(base, 'Data*'))):
    files = sorted(glob.glob(os.path.join(d, '*.dcm')))
    if not files:
        continue
    try:
        ds = pydicom.dcmread(files[0], force=True, stop_before_pixels=True)
        series_desc = ds.get('SeriesDescription', 'N/A')
        modality = ds.get('Modality', 'N/A')
        series_num = ds.get('SeriesNumber', 'N/A')
        num_frames = ds.get('NumberOfFrames', 'N/A')
        rows = ds.get('Rows', 'N/A')
        cols = ds.get('Columns', 'N/A')
        nf = len(files)
        print(f'{os.path.basename(d)}: Modality={modality}, Series={series_num}, Desc={series_desc}, Rows={rows}, Cols={cols}, Files={nf}, Frames={num_frames}')
    except Exception as e:
        print(f'{os.path.basename(d)}: Error - {e}')
