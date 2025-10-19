import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

const _kDisclaimerAcceptedKey = 'disclaimerAcceptedAt'; // stores ISO timestamp

class DisclaimerGate {
  /// Call this before navigating to the AI Check screen.
  /// Set [alwaysShow] = true if you want the popup every time (ignores saved consent).
  static Future<bool> ensureAccepted(BuildContext context, {bool alwaysShow = false}) async {
    final prefs = await SharedPreferences.getInstance();
    final alreadyAccepted = prefs.containsKey(_kDisclaimerAcceptedKey);

    if (alreadyAccepted && !alwaysShow) return true;

    final accepted = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => const _DisclaimerDialog(),
    );

    if (accepted == true) {
      await prefs.setString(_kDisclaimerAcceptedKey, DateTime.now().toUtc().toIso8601String());
      return true;
    }
    return false;
  }
}

class _DisclaimerDialog extends StatefulWidget {
  const _DisclaimerDialog({Key? key}) : super(key: key);

  @override
  State<_DisclaimerDialog> createState() => _DisclaimerDialogState();
}

class _DisclaimerDialogState extends State<_DisclaimerDialog> {
  bool _checked = false;

  Future<void> _openPolicy() async {
    final uri = Uri.parse('https://www.ironstronginitiative.com/privacy');
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      // ignore: use_build_context_synchronously
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Could not open Privacy Policy')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Important: Not a Medical Device'),
      content: SizedBox(
        width: 380,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text(
                'ISHI provides an AI-assisted anemia risk indication from an eyelid photo. '
                'It is for awareness only and is NOT a diagnosis or a substitute for professional medical advice.',
              ),
              SizedBox(height: 12),
              Text('By continuing, you acknowledge that:'),
              SizedBox(height: 6),
              _Bullet('Results may be inaccurate and can be affected by lighting, camera quality, and positioning.'),
              _Bullet('You will not rely on this app to make medical decisions.'),
              _Bullet('If you have symptoms or concerns, you will consult a qualified clinician.'),
              SizedBox(height: 12),
              Text('See our Privacy Policy for how your data is handled.'),
            ],
          ),
        ),
      ),
      actionsAlignment: MainAxisAlignment.spaceBetween,
      actions: [
        TextButton(
          onPressed: _openPolicy,
          child: const Text('Privacy Policy'),
        ),
        Row(
          children: [
            Checkbox(
              value: _checked,
              onChanged: (v) => setState(() => _checked = v ?? false),
            ),
            const Text('I have read and agree'),
            const SizedBox(width: 12),
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            const SizedBox(width: 8),
            ElevatedButton(
              onPressed: _checked ? () => Navigator.of(context).pop(true) : null,
              child: const Text('Continue'),
            ),
          ],
        ),
      ],
    );
  }
}

class _Bullet extends StatelessWidget {
  final String text;
  const _Bullet(this.text, {Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('•  '),
        Expanded(child: Text(text)),
      ],
    );
  }
}
