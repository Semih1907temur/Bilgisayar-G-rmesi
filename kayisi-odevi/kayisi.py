import os
import torch
from ultralytics import YOLO
from roboflow import Roboflow

# CUDA ve GPU kullanılabilirliğini kontrol et
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

# Roboflow API anahtarı ile proje verilerini indir
rf = Roboflow(api_key="RCSieCKbpY7YeHdrGxO4")
project = rf.workspace("semih19074").project("kayisi-tespiti")
version = project.version(2)
dataset = version.download("yolov8")

# Kayıt dizinini tanımla
results_dir = os.path.join(dataset.location, "results")
os.makedirs(results_dir, exist_ok=True)  # "results" klasörü yoksa oluştur

# Eğitim işlemi ve etiketli görüntüleri kaydetme
if __name__ == "__main__":
    # Modeli eğit
    model.train(data=f"{dataset.location}/data.yaml", epochs=10, imgsz=640)

    # Eğitim sonrası test verilerinden bazı görüntüleri modelin tahmin etmesi ve kaydetmesi için
    test_images = os.listdir(dataset.location + "/test/images")[:10]  # İlk 10 test görüntüsünü al

    for idx, image_name in enumerate(test_images):
        image_path = os.path.join(dataset.location, "test", "images", image_name)
        results = model.predict(image_path)  # Görüntü üzerinde tahmin yap

        # Tahmin edilen görüntüyü kaydet
        results[0].plot()  # Görüntüyü etiketlerle göster
        save_path = os.path.join(results_dir, f"labelled_image_{idx}.jpg")
        results[0].save(save_path)  # Etiketli görüntüyü kaydet

print("Eğitim tamamlandı ve etiketlenmiş test görüntüleri 'results' klasörüne kaydedildi.")

# C:\Users\semih\PycharmProjects\semih1\Kayısı-Tespiti-2\results
