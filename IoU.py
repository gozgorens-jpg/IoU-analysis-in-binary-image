import numpy as np
from qgis.core import QgsProject

# Katman adları
layer_name_1 = '"predicted/classified layer"'
layer_name_2 = '"reference layer/ ground truth"'

# Katmanları al
layer1 = QgsProject.instance().mapLayersByName(layer_name_1)[0]
layer2 = QgsProject.instance().mapLayersByName(layer_name_2)[0]

# Raster verilerini numpy array'e çevir
provider1 = layer1.dataProvider()
provider2 = layer2.dataProvider()

# Raster boyutlarını kontrol et
extent1 = layer1.extent()
extent2 = layer2.extent()
width1 = layer1.width()
height1 = layer1.height()
width2 = layer2.width()
height2 = layer2.height()

print(f"\n{layer_name_1} boyutları: {width1} x {height1}")
print(f"{layer_name_2} boyutları: {width2} x {height2}")

# İlk katmanın band 1 verilerini oku
block1 = provider1.block(1, extent1, width1, height1)
arr1 = np.zeros((height1, width1))
for i in range(height1):
    for j in range(width1):
        arr1[i, j] = block1.value(i, j)

# İkinci katmanın band 1 verilerini oku
block2 = provider2.block(1, extent2, width2, height2)
arr2 = np.zeros((height2, width2))
for i in range(height2):
    for j in range(width2):
        arr2[i, j] = block2.value(i, j)

# Binary map'lere dönüştür (0 ve 1 değerleri)
# NoData değerlerini filtrele
nodata1 = provider1.sourceNoDataValue(1)
nodata2 = provider2.sourceNoDataValue(1)

# Binary'ye dönüştür (1 = flood, 2 = non-flood)
# Flood piksellerini (değer=1) -> 1'e, diğer her şeyi (değer=2 ve NoData) -> 0'a dönüştür
binary1 = np.where((arr1 == 1), 1, 0)
binary2 = np.where((arr2 == 1), 1, 0)

# Boyutlar uyuşmuyorsa uyarı ver
if binary1.shape != binary2.shape:
    print("\n!!! UYARI: Katmanların boyutları farklı!")
    print(f"Minimum boyuta göre kırpılacak...")
    min_height = min(height1, height2)
    min_width = min(width1, width2)
    binary1 = binary1[:min_height, :min_width]
    binary2 = binary2[:min_height, :min_width]

# IoU Hesaplama
intersection = np.logical_and(binary1, binary2).sum()
union = np.logical_or(binary1, binary2).sum()

# Union sıfır ise (iki map de tamamen boş)
if union == 0:
    iou = 0.0
    print("\n!!! UYARI: Her iki katman da boş (union = 0)")
else:
    iou = intersection / union

# Diğer metrikler
total_pixels = binary1.size
true_positive = intersection  # Her iki map'te de 1 olan pikseller
false_positive = np.logical_and(binary2 == 1, binary1 == 0).sum()  # Sadece 2'de 1
false_negative = np.logical_and(binary1 == 1, binary2 == 0).sum()  # Sadece 1'de 1
true_negative = np.logical_and(binary1 == 0, binary2 == 0).sum()   # Her ikisi de 0

# Precision, Recall ve F1-Score
if (true_positive + false_positive) > 0:
    precision = true_positive / (true_positive + false_positive)
else:
    precision = 0.0

if (true_positive + false_negative) > 0:
    recall = true_positive / (true_positive + false_negative)
else:
    recall = 0.0

if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0

# Overall Accuracy
overall_accuracy = (true_positive + true_negative) / total_pixels

# Sonuçları yazdır
print("\n" + "="*60)
print("BINARY MAP DOĞRULUK ANALİZİ SONUÇLARI")
print("="*60)
print(f"\nKatman 1 (Referans): {layer_name_1}")
print(f"Katman 2 (Tahmin):   {layer_name_2}")
print(f"\nToplam Piksel Sayısı: {total_pixels:,}")
print("\n--- Confusion Matrix ---")
print(f"True Positive  (TP): {true_positive:,}")
print(f"False Positive (FP): {false_positive:,}")
print(f"False Negative (FN): {false_negative:,}")
print(f"True Negative  (TN): {true_negative:,}")
print("\n--- Doğruluk Metrikleri ---")
print(f"IoU (Intersection over Union): {iou:.4f} ({iou*100:.2f}%)")
print(f"Precision (Hassasiyet):        {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall (Duyarlılık):           {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:                      {f1_score:.4f} ({f1_score*100:.2f}%)")
print(f"Overall Accuracy (Genel Doğr.):{overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print("\n--- Alan Analizi ---")
print(f"{layer_name_1} 'flood' piksel sayısı: {binary1.sum():,}")
print(f"{layer_name_2} 'flood' piksel sayısı: {binary2.sum():,}")
print(f"Kesişim (Intersection): {intersection:,}")
print(f"Birleşim (Union):       {union:,}")
print("="*60)

# İsteğe bağlı: Sonuçları dosyaya kaydet
results_text = f"""
BINARY MAP DOĞRULUK ANALİZİ SONUÇLARI
{"="*60}

Katman 1 (Referans): {layer_name_1}
Katman 2 (Tahmin):   {layer_name_2}

Toplam Piksel Sayısı: {total_pixels:,}

Confusion Matrix:
True Positive  (TP): {true_positive:,}
False Positive (FP): {false_positive:,}
False Negative (FN): {false_negative:,}
True Negative  (TN): {true_negative:,}

Doğruluk Metrikleri:
IoU (Intersection over Union): {iou:.4f} ({iou*100:.2f}%)
Precision (Hassasiyet):        {precision:.4f} ({precision*100:.2f}%)
Recall (Duyarlılık):           {recall:.4f} ({recall*100:.2f}%)
F1-Score:                      {f1_score:.4f} ({f1_score*100:.2f}%)
Overall Accuracy (Genel Doğr.):{overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)

Alan Analizi:
{layer_name_1} 'flood' piksel sayısı: {binary1.sum():,}
{layer_name_2} 'flood' piksel sayısı: {binary2.sum():,}
Kesişim (Intersection): {intersection:,}
Birleşim (Union):       {union:,}
{"="*60}
"""

# Dosyaya kaydet (isteğe bağlı - yorum satırını kaldırarak aktifleştirebilirsiniz)
# with open('/path/to/iou_results.txt', 'w', encoding='utf-8') as f:
#     f.write(results_text)
# print("\nSonuçlar dosyaya kaydedildi: /path/to/iou_results.txt")