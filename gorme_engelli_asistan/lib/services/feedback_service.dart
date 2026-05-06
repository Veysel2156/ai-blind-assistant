import 'package:flutter_tts/flutter_tts.dart';
import 'package:vibration/vibration.dart';

class FeedbackService {
  final FlutterTts _flutterTts = FlutterTts();

  FeedbackService() {
    _initTts();
  }

  void _initTts() async {
    await _flutterTts.setLanguage("tr-TR");
    await _flutterTts.setSpeechRate(0.5);
    await _flutterTts.setPitch(1.0);
  }

  Future<void> speak(String text) async {
    await _flutterTts.speak(text);
  }

  Future<void> vibrate() async {
    bool? hasVibrator = await Vibration.hasVibrator();
    if (hasVibrator == true) {
      Vibration.vibrate(duration: 150);
    }
  }

  void stop() async {
    await _flutterTts.stop();
  }
}