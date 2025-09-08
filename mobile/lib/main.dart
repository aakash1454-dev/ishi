import 'package:flutter/material.dart';
import 'pages/home_page.dart';

void main() {
  runApp(const ISHIApp());
}

class ISHIApp extends StatelessWidget {
  const ISHIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ISHI App',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        scaffoldBackgroundColor: Colors.white,
      ),
      home: const HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}