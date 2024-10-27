
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import torch

# YOLO modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Kullanacağımız görüntü URL'leri
image_urls = [
    "https://cdn.pixabay.com/photo/2017/05/22/07/20/press-2333329_640.jpg",
    "https://cdn.pixabay.com/photo/2020/04/22/05/25/grand-central-terminal-5075970_640.jpg",
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1t07Df.img?w=460&h=340&m=6",
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1sZtwg.img?w=768&h=432&m=6&x=520&y=111&s=102&d=102"
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1t0nqK.img?w=460&h=340&m=6",
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1t0Dmo.img?w=768&h=764&m=6&x=451&y=558&s=651&d=256"
    "https://www.bmw.com.tr/content/dam/bmw/common/all-models/3-series/sedan/2022/highlights/bmw-3-series-sedan-sp-desktop.jpg/jcr:content/renditions/cq5dam.resized.img.1680.large.time1651158408008.jpg"
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1t070C.img?w=750&h=422&m=6&x=229&y=55&s=290&d=183"
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1sYDqa.img?w=750&h=422&m=6&x=316&y=116&s=155&d=155"
    "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1t07Df.img?w=460&h=340&m=6"
]

# Sonuçları saklamak için liste
results_data = []

for url in image_urls:
    try:
        # Görüntüyü URL'den indir
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # YOLO ile tahmin
        results = model(img)

        # Sonuçları al
        detected_objects = results.pandas().xyxy[0]
        labels = detected_objects['name'].tolist()

        # İnsan olup olmadığını kontrol et
        if 'person' in labels:
            classification = "İnsan var"
        else:
            classification = "İnsan yok"

        # Sonuçları kaydet
        results_data.append({
            "Görüntü URL": url,
            "Sonuç": classification
        })

        # Sonucu ekrana yazdır
        print(f"{url}: {classification}")

    except Exception as e:
        print(f"{url} indirilemedi: {e}")

# Sonuçları bir CSV dosyasına kaydet
df = pd.DataFrame(results_data)
df.to_csv("sınıflandırma_sonuçları.csv", index=False)
print("Sonuçlar sınıflandırma_sonuçları.csv dosyasına kaydedildi.")
