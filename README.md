# 🔊 Görme Engelliler İçin Sesli Nesne Tanıma (Edge AI Assistant)

![Flutter](https://img.shields.io/badge/Flutter-%2302569B.svg?style=for-the-badge&logo=Flutter&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

Bu proje, görme engelli bireylerin ev ortamında karşılaştıkları nesneleri bağımsız olarak tespit edebilmeleri için geliştirilmiş, **tamamen çevrimdışı (Edge AI)** çalışan bir mobil asistan uygulamasıdır. 

## 📌 Proje Özeti
Sistem, bulut (cloud) API'lerine bağımlı kalmadan cihazın kendi işlemcisi üzerinde çalışan INT8 formatında kuantize edilmiş bir nesne tespiti modeline dayanmaktadır. Uygulama, tespit edilen nesneleri Türkçe olarak seslendirir (TTS) ve dokunsal geri bildirim sağlar.

- **Yapay Zeka Mimarı:** YOLOv8 (MobileNetV2 Altyapısı ile Transfer Learning)
- **Model Formatı:** TFLite (INT8 Post-Training Quantization - Yalnızca 2.7 MB)
- **Mobil Çerçeve:** Flutter (Android)
- **Geri Bildirim:** flutter_tts (Türkçe) & Titreşim Motoru (Haptic)
- **Doğruluk Oranı:** Test veri setinde **%88 Accuracy**
- **Çıkarım Hızı (Latency):** Gerçek zamanlı (Ortalama 59 ms)

## 🏷️ Desteklenen Nesneler (13 Sınıf)
Klima, Şişe, Kase, Saat, Bardak, Çatal, Bıçak, Kalem, İnsan, Telefon, Makas, Kaşık, Çaydanlık.

## ⚙️ Performans ve Mimari Özellikleri
- **3-Frame Ortalaması:** Kameradaki mikro titremelerin yarattığı "kararsız tahminleri" filtreler.
- **Center Crop (Merkez Kırpma):** Yalnızca hedeflenen merkez kareyi işleyerek arka plan gürültüsünü engeller.
- **Asenkron Mutex Kilidi:** Yüksek FPS'den kaynaklı RAM (Buffer) taşmalarını önler, eski tahminleri debouncing ile korur.

## 🚀 Kurulum (Local Development)

Proje kaynak kodlarını cihazınıza indirdikten sonra çalıştırmak için:

```bash
# Depoyu klonlayın
git clone [https://github.com/Veysel2156/ai-blind-assistant.git](https://github.com/Veysel2156/ai-blind-assistant.git)

# Klasöre girin
cd ai-blind-assistant

# Gerekli Flutter paketlerini yükleyin
flutter pub get

# USB Hata Ayıklama modunda cihazınıza kurun
flutter run
