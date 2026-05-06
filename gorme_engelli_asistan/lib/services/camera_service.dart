import 'package:camera/camera.dart';

class CameraService {
  CameraController? cameraController;
  int _frameCount = 0;
  bool isProcessing = false;

  Future<void> initialize() async {
    final cameras = await availableCameras();
    final backCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
    );

    cameraController = CameraController(
      backCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await cameraController!.initialize();
  }

  void startImageStream(Function(CameraImage) onProcessFrame) {
    if (cameraController == null || !cameraController!.value.isInitialized) return;

    cameraController!.startImageStream((CameraImage image) {
      _frameCount++;

      if (_frameCount % 5 == 0 && !isProcessing) {
        isProcessing = true;
        onProcessFrame(image);
        _frameCount = 0;
      }
    });
  }

  void stopImageStream() {
    if (cameraController != null && cameraController!.value.isStreamingImages) {
      cameraController!.stopImageStream();
    }
  }

  void dispose() {
    cameraController?.dispose();
  }
}