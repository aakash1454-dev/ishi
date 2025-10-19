import 'package:flutter/material.dart';
import 'pages/home_page.dart';
import 'pages/camera_page.dart'; // pushed after disclaimer
import 'pages/profile_page.dart';
import 'widgets/floating_nav.dart';
import 'pages/about_page.dart';
import 'widgets/disclaimer_gate.dart';

void main() => runApp(const ISHIApp());

class ISHIApp extends StatefulWidget {
  const ISHIApp({super.key});
  @override
  State<ISHIApp> createState() => _ISHIAppState();
}

class _ISHIAppState extends State<ISHIApp> {
  // NEW: navigator key so dialogs always have a proper Navigator context
  final GlobalKey<NavigatorState> _navKey = GlobalKey<NavigatorState>();
  int _index = 0;

  // Intercept AI Check (index 1): show disclaimer, then push CameraPage
  Future<void> _handleNavTap(int i) async {
    const aiCheckIndex = 1; // matches FloatingNavBar's "ISHI-AI Check"

    if (i == aiCheckIndex) {
      final ctx = _navKey.currentContext ?? context;

      // Tiny toast so you know the tap fired
      ScaffoldMessenger.of(ctx).showSnackBar(
        const SnackBar(
          content: Text('Opening ISHI-AI Check…'),
          duration: Duration(milliseconds: 500),
        ),
      );

      // Show the disclaimer using a Navigator-aware context
      final ok = await DisclaimerGate.ensureAccepted(ctx, alwaysShow: true);
      if (!ok) return;

      // Navigate only after acceptance
      await _navKey.currentState!.push(
        MaterialPageRoute(builder: (_) => const CameraPage()),
      );
      return; // do not switch the tab highlight
    }

    // Normal tab switch
    setState(() => _index = i);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ISHI App',
      debugShowCheckedModeBanner: false,
      navigatorKey: _navKey, // <-- IMPORTANT
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
            // Keep indices aligned with FloatingNavBar items (0..5)
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

// Placeholders
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
