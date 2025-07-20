import 'package:flutter/material.dart';
import 'pages/home_page.dart';

void main() {
  runApp(const ISHIApp());
}

class ISHIApp extends StatelessWidget {
  const ISHIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const HomePage();
  }
}
