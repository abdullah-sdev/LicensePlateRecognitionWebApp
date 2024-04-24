import torch

# Model
# model = torch.hub.load("yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
# Images
img = "D:\Projects\Pythons\FaceRecognotion_c4.0\TestDataRecording\DataSet-CarLicensePlate\DataSet-CarLicensePlate\DSC_0329.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.