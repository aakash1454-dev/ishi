import 'package:flutter/material.dart';
import 'pages/home_page.dart';
import 'pages/camera_page.dart'; // ISHI-AI Check
import 'widgets/floating_nav.dart';

void main() => runApp(const ISHIApp());

class ISHIApp extends StatefulWidget {
  const ISHIApp({super.key});
  @override
  State<ISHIApp> createState() => _ISHIAppState();
}

class _ISHIAppState extends State<ISHIApp> {
  int _index = 0;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ISHI App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: const Color(0xFF2B5CFF)),
      darkTheme: ThemeData(useMaterial3: true, colorSchemeSeed: const Color(0xFF2B5CFF), brightness: Brightness.dark),
      home: Scaffold(
        body: SafeArea(
          child: IndexedStack(
            index: _index,
            children: const [
              HomePage(),
              CameraPage(),          // ISHI-AI Check
              _EventsPage(),
              _ProfilePage(),
              _AboutPage(),
              _DonatePage(),
            ],
          ),
        ),
        bottomNavigationBar: FloatingNavBar(
          currentIndex: _index,
          onTap: (i) => setState(() => _index = i),
        ),
      ),
    );
  }
}

// Lightweight placeholders (you can expand later)
class _EventsPage extends StatelessWidget {
  const _EventsPage();
  @override
  Widget build(BuildContext context) => const Center(child: Text('Events coming soon'));
}
class _ProfilePage extends StatelessWidget {
  const _ProfilePage();
  @override
  Widget build(BuildContext context) => const Center(child: Text('Profile coming soon'));
}
class _AboutPage extends StatelessWidget {
  const _AboutPage();
  @override
  Widget build(BuildContext context) => const Center(child: Text('About ISHI'));
}
class _DonatePage extends StatelessWidget {
  const _DonatePage();
  @override
  Widget build(BuildContext context) => const Center(child: Text('Donate link / QR here'));
}
