import 'package:flutter/material.dart';
import 'pages/home_page.dart';
import 'pages/camera_page.dart'; // pushed after disclaimer
import 'pages/profile_page.dart';
import 'widgets/floating_nav.dart';
import 'pages/about_page.dart';
// import 'package:isar_flutter_libs/isar_flutter_libs.dart' as _;

// NEW: disclaimer gate
import 'widgets/disclaimer_gate.dart';

void main() => runApp(const ISHIApp());

class ISHIApp extends StatefulWidget {
  const ISHIApp({super.key});
  @override
  State<ISHIApp> createState() => _ISHIAppState();
}

class _ISHIAppState extends State<ISHIApp> {
  int _index = 0;

  // Intercept AI Check (index 1): show disclaimer, then push CameraPage
  Future<void> _handleNavTap(int i) async {
    const aiCheckIndex = 1; // matches FloatingNavBar's "ISHI-AI Check"

    if (i == aiCheckIndex) {
      final ok = await DisclaimerGate.ensureAccepted(context, alwaysShow: true);
      if (!ok || !mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => const CameraPage()),
      );
      return; // don't switch the tab highlight
    }

    // Normal tab switch
    setState(() => _index = i);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ISHI App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: const Color(0xFF2B5CFF)),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF2B5CFF),
        brightness: Brightness.dark,
      ),
      home: Scaffold(
        body: SafeArea(
          child: IndexedStack(
            index: _index,
            // IMPORTANT: Keep indices aligned with FloatingNavBar items (0..5)
            children: const [
              HomePage(),             // 0
              _AiCheckPlaceholder(),  // 1 (AI Check handled via push, not a tab)
              _EventsPage(),          // 2
              ProfilePage(),          // 3
              AboutPage(),            // 4
              _DonatePage(),          // 5
            ],
          ),
        ),
        bottomNavigationBar: FloatingNavBar(
          currentIndex: _index,
          onTap: _handleNavTap,
        ),
      ),
    );
  }
}

// Lightweight placeholders
class _AiCheckPlaceholder extends StatelessWidget {
  const _AiCheckPlaceholder();
  @override
  Widget build(BuildContext context) => const SizedBox.shrink();
}

class _EventsPage extends StatelessWidget {
  const _EventsPage();
  @override
  Widget build(BuildContext context) =>
      const Center(child: Text('Events coming soon'));
}

class _DonatePage extends StatelessWidget {
  const _DonatePage();
  @override
  Widget build(BuildContext context) =>
      const Center(child: Text('Donate link / QR here'));
}
