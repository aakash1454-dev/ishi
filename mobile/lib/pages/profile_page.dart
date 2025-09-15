// lib/pages/profile_page.dart
import 'package:flutter/material.dart';
import '../services/auth/google_auth.dart';
import '../services/storage/store.dart';


class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  final _auth = GoogleAuthService();

  bool _signedIn = false;
  String? _name, _email, _photo;
  String _tokenPreview = 'Tap to fetch ID token';
  List<TestResult> _history = [];

  @override
  void initState() {
    super.initState();
    _refreshProfile();
    _loadHistory();
  }

  Future<void> _refreshProfile() async {
    _name = await _auth.name;
    _email = await _auth.email;
    _photo = await _auth.photo;
    setState(() {
      _signedIn = (_email != null && _email!.isNotEmpty);
    });
  }

  Future<void> _loadHistory() async {
    final items = await LocalStore.listResults(limit: 200);
    if (!mounted) return;
    setState(() => _history = items);
  }

  Future<void> _signIn() async {
    final acct = await _auth.signIn();
    if (acct != null) {
      await _refreshProfile();
    }
  }

  Future<void> _signOut() async {
    // Wipe local secure storage so UI reflects sign-out correctly.
    await _auth.signOut(wipeLocal: true);
    await _refreshProfile();
  }

  Future<void> _fetchIdToken() async {
    final token = await _auth.idToken;
    if (token == null || token.isEmpty) return;
    setState(() {
      _tokenPreview = token.length > 18 ? '${token.substring(0, 18)}…' : token;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 480),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Card(
                  elevation: 2,
                  child: Padding(
                    padding: const EdgeInsets.all(18),
                    child: _signedIn ? _signedInView() : _signedOutView(),
                  ),
                ),
                const SizedBox(height: 16),
                if (_signedIn) _historyView(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _signedOutView() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text(
          'Sign in to personalize ISHI on this device. Your data stays on-device.',
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: _signIn,
          icon: const Icon(Icons.login),
          label: const Text('Continue with Google'),
        ),
        const SizedBox(height: 8),
        const Text(
          'No cloud storage. You can sign out anytime.',
          style: TextStyle(fontSize: 12),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _signedInView() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        CircleAvatar(
          radius: 36,
          backgroundImage:
              (_photo != null && _photo!.isNotEmpty) ? NetworkImage(_photo!) : null,
          child: (_photo == null || _photo!.isEmpty)
              ? const Icon(Icons.person, size: 36)
              : null,
        ),
        const SizedBox(height: 10),
        Text(
          _name ?? 'Anonymous',
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 4),
        Text(_email ?? '', style: const TextStyle(color: Colors.grey)),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          alignment: WrapAlignment.center,
          children: [
            FilledButton.tonal(
              onPressed: _fetchIdToken,
              child: Text(_tokenPreview),
            ),
            FilledButton.tonal(
              onPressed: _signOut,
              child: const Text('Sign out'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _historyView() {
    if (_history.isEmpty) {
      return const Text('No local test history yet.');
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Local Test History', style: TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 8),
        ListView.separated(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: _history.length,
          itemBuilder: (ctx, i) {
            final r = _history[i];
            return ListTile(
              leading: Icon(r.anemic ? Icons.warning_amber : Icons.check_circle_outline),
              title: Text(r.anemic ? 'Anemic' : 'Not Anemic'),
              subtitle: Text(
                '${r.timestamp.toLocal()}  •  score: ${r.score.toStringAsFixed(2)}',
              ),
            );
          },
          separatorBuilder: (_, __) => const Divider(height: 1),
        ),
      ],
    );
  }
}
