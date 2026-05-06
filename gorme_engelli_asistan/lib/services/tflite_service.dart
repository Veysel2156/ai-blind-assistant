import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart';

class TFLiteService {
  Interpreter? _interpreter;
  List<String> _labels = [];

  // İngilizce → Türkçe çeviri tablosu
  static const Map<String, String> _turkishNames = {
    'air-conditioner': 'Klima',
    'bottle'         : 'Şişe',
    'bowl'           : 'Kase',
    'clock'          : 'Saat',
    'cup'            : 'Bardak',
    'fork'           : 'Çatal',
    'knife'          : 'Bıçak',
    'pen'            : 'Kalem',
    'person'         : 'İnsan',
    'phone'          : 'Telefon',
    'scissor'        : 'Makas',
    'spoon'          : 'Kaşık',
    'teapot'         : 'Çaydanlık',
  };

  // Sabit tamponlar (GC baskısını azaltır)
  Int8List?        _inputBuffer;
  List<List<int>>? _outputBuffer;

  // Multi-frame ortalama için skor geçmişi
  final List<List<double>> _scoreHistory = [];
  static const int _historySize = 3;

  double _outScale     = 1.0;
  int    _outZeroPoint = 0;
  int    _numClasses   = 0;

  static const int    inputSize = 224;
  static const double threshold = 0.22;

  Future<void> loadModel() async {
    try {
      final options = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromAsset(
        'assets/model_optimized_int8.tflite',
        options: options,
      );

      final inputTensor  = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);

      _numClasses   = outputTensor.shape.last;
      _outScale     = outputTensor.params.scale;
      _outZeroPoint = outputTensor.params.zeroPoint;

      _inputBuffer  = Int8List(inputSize * inputSize * 3);
      _outputBuffer = [List<int>.filled(_numClasses, 0)];

      print('=== MODEL YÜKLENDI ===');
      print('Input : ${inputTensor.type}  ${inputTensor.shape}');
      print('Output: ${outputTensor.type}  ${outputTensor.shape}  scale=$_outScale  zp=$_outZeroPoint');

      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();
      print('Etiket sayısı: ${_labels.length}');
    } catch (e) {
      print('Model yükleme hatası: $e');
    }
  }

  /// İngilizce etiketi Türkçeye çevirir
  String _toTurkish(String englishLabel) {
    return _turkishNames[englishLabel.toLowerCase()] ?? englishLabel;
  }

  Future<String?> processCameraImage(CameraImage image) async {
    if (_interpreter == null || _inputBuffer == null || _outputBuffer == null) return null;
    try {
      // Merkez kırpma + 90° CW + int8
      _centerCropToInt8(image, _inputBuffer!);
      final input = _inputBuffer!.reshape([1, inputSize, inputSize, 3]);

      for (int i = 0; i < _numClasses; i++) _outputBuffer![0][i] = 0;
      _interpreter!.run(input, _outputBuffer!);

      // Bu frame'in dequantize edilmiş skorları
      final frameScores = List<double>.generate(
        _numClasses,
        (i) => (_outputBuffer![0][i] - _outZeroPoint) * _outScale,
      );

      // Geçmişe ekle
      _scoreHistory.add(frameScores);
      if (_scoreHistory.length > _historySize) _scoreHistory.removeAt(0);

      // Ortalama skor hesapla
      final avgScores = List<double>.filled(_numClasses, 0.0);
      for (final s in _scoreHistory) {
        for (int i = 0; i < _numClasses; i++) avgScores[i] += s[i];
      }
      for (int i = 0; i < _numClasses; i++) avgScores[i] /= _scoreHistory.length;

      // En yüksek skoru bul
      double maxScore = -double.infinity;
      int    maxIdx   = 0;
      for (int i = 0; i < _numClasses; i++) {
        if (avgScores[i] > maxScore) {
          maxScore = avgScores[i];
          maxIdx   = i;
        }
      }

      if (maxScore >= threshold && maxIdx < _labels.length) {
        final englishLabel = _labels[maxIdx];
        final turkishLabel = _toTurkish(englishLabel);
        final pct          = (maxScore * 100).toStringAsFixed(0);
        print('Tahmin: $turkishLabel ($englishLabel) → $pct%');
        // Ekran metni: Türkçe | TTS için sadece Türkçe isim döner
        return '$turkishLabel (%$pct)';
      }
      return null;
    } catch (e) {
      print('İşleme hatası: $e');
      return null;
    }
  }

  /// Ekrana gösterilecek metin Türkçe, TTS için de Türkçe isim al
  String getTurkishName(String resultText) {
    // "Şişe (%85)" → "Şişe"
    return resultText.split(' ').first;
  }

  /// Merkez kare kırpma + 90° CW döndürme + int8 normalizasyon
  void _centerCropToInt8(CameraImage image, Int8List out) {
    final imgW = image.width;
    final imgH = image.height;

    final cropSize = min(imgW, imgH);
    final startX   = (imgW - cropSize) ~/ 2;
    final startY   = (imgH - cropSize) ~/ 2;

    final yBytes   = image.planes[0].bytes;
    final uBytes   = image.planes[1].bytes;
    final vBytes   = image.planes[2].bytes;
    final yStride  = image.planes[0].bytesPerRow;
    final uvStride = image.planes[1].bytesPerRow;
    final uvPixel  = image.planes[1].bytesPerPixel ?? 1;

    int idx = 0;
    for (int outY = 0; outY < inputSize; outY++) {
      for (int outX = 0; outX < inputSize; outX++) {
        final cropX = (outY * cropSize ~/ inputSize).clamp(0, cropSize - 1);
        final cropY = (cropSize - 1) - (outX * cropSize ~/ inputSize).clamp(0, cropSize - 1);

        final srcX = (startX + cropX).clamp(0, imgW - 1);
        final srcY = (startY + cropY).clamp(0, imgH - 1);

        final yVal = yBytes[srcY * yStride + srcX] & 0xFF;
        final uvIdx = (srcY ~/ 2) * uvStride + (srcX ~/ 2) * uvPixel;
        final u = (uBytes[uvIdx] & 0xFF) - 128;
        final v = (vBytes[uvIdx] & 0xFF) - 128;

        out[idx++] = (yVal + 1.402 * v).clamp(0, 255).toInt() - 128;
        out[idx++] = (yVal - 0.344136 * u - 0.714136 * v).clamp(0, 255).toInt() - 128;
        out[idx++] = (yVal + 1.772 * u).clamp(0, 255).toInt() - 128;
      }
    }
  }

  void resetHistory() => _scoreHistory.clear();

  void dispose() {
    _interpreter?.close();
  }
}