import 'package:flutter/material.dart';
import 'ui/home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const GormeEngelliAsistanApp());
}

class GormeEngelliAsistanApp extends StatelessWidget {
  const GormeEngelliAsistanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sesli Nesne Tanıma',
      debugShowCheckedModeBanner: false,
      home: HomeScreen(),
    );
  }
}