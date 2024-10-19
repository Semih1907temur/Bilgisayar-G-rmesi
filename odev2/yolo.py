import torch

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Resmi yükle
img = 'C:\python-images/1.jpg'

# Resmi tahmin et
results = model(img)

# Sonuçları görselleştir
results.show()

# Sonuçları yazdır
results.print()
