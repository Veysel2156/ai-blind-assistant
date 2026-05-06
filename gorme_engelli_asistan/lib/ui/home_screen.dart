import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/tflite_service.dart';
import '../services/feedback_service.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  CameraController? _cameraController;
  bool _isDetecting = false;
  bool _isCameraPaused = false;
  String _predictionText = "Nesne Aranıyor...";
  bool _hasDetection = false; // Son tahmin başarılı mıydı?
  int _frameCount = 0;         // Frame atlama sayacı

  final TFLiteService    _tfLiteService    = TFLiteService();
  final FeedbackService  _feedbackService  = FeedbackService();

  @override
  void initState() {
    super.initState();
    _initializeCameraAndModel();
  }

  Future<void> _initializeCameraAndModel() async {
    await _tfLiteService.loadModel();

    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.low, // low → daha az bellek, daha hızlı frame işleme
      enableAudio: false,
    );

    await _cameraController!.initialize();
    if (!mounted) return;
    setState(() {});

    _cameraController!.startImageStream((CameraImage image) async {
      // Her 4 frame'de 1 işle (performans iyileştirme)
      _frameCount++;
      if (_frameCount % 4 != 0) return;

      if (_isCameraPaused || _isDetecting) return;
      _isDetecting = true;

      final String? result = await _tfLiteService.processCameraImage(image);

      if (result != null && mounted) {
        // Yeni veya farklı nesne → sesli bildir
        if (result != _predictionText || !_hasDetection) {
          final turkishName = _tfLiteService.getTurkishName(result);
          _feedbackService.speak(turkishName); // "Şişe", "Kalem" gibi
          _feedbackService.vibrate();
        }
        setState(() {
          _predictionText = result;
          _hasDetection = true;
        });
        await Future.delayed(const Duration(milliseconds: 1500));
      } else if (mounted && _hasDetection) {
        // Eşik altına düştü — son sonucu 2 saniye daha göster, sonra sıfırla
        await Future.delayed(const Duration(seconds: 2));
        if (mounted && !_isCameraPaused) {
          setState(() {
            _predictionText = "Nesne Aranıyor...";
            _hasDetection = false;
          });
        }
      }

      _isDetecting = false;
    });
  }

  void _toggleCameraState() {
    setState(() {
      _isCameraPaused = !_isCameraPaused;
      if (_isCameraPaused) {
        _feedbackService.stop();
        _predictionText = "Kamera Duraklatıldı";
        _hasDetection = false;
      } else {
        _predictionText = "Nesne Aranıyor...";
      }
    });
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _tfLiteService.dispose();
    _feedbackService.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text("Yapay Zeka Asistanı"),
        backgroundColor: Colors.blueGrey[900],
      ),
      body: Stack(
        children: [
          // 1. Kamera Görüntüsü
          if (_cameraController != null && _cameraController!.value.isInitialized)
            Positioned.fill(
              child: _isCameraPaused
                  ? const Center(
                      child: Text(
                        "Kamera Kapalı",
                        style: TextStyle(color: Colors.white54, fontSize: 24),
                      ),
                    )
                  : CameraPreview(_cameraController!),
            )
          else
            const Center(child: CircularProgressIndicator()),

          // 2. Üst Kısım: Tahmin Sonucu
          Positioned(
            top: 20,
            left: 16,
            right: 16,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
              decoration: BoxDecoration(
                color: _hasDetection
                    ? Colors.green.withOpacity(0.85)
                    : Colors.black.withOpacity(0.7),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: _hasDetection ? Colors.greenAccent : Colors.white24,
                  width: 2,
                ),
              ),
              child: Text(
                _predictionText,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  shadows: _hasDetection
                      ? [const Shadow(color: Colors.black54, blurRadius: 4)]
                      : null,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),

          // 3. Alt Kısım: Aç/Kapa Butonu
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 50.0),
              child: GestureDetector(
                onTap: _toggleCameraState,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  height: 90,
                  width: 90,
                  decoration: BoxDecoration(
                    color: _isCameraPaused ? Colors.green : Colors.redAccent,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: _isCameraPaused
                            ? Colors.green.withOpacity(0.5)
                            : Colors.redAccent.withOpacity(0.5),
                        blurRadius: 15,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: Icon(
                    _isCameraPaused ? Icons.play_arrow_rounded : Icons.pause_rounded,
                    color: Colors.white,
                    size: 60,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}