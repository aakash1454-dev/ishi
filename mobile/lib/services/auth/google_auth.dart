import 'package:google_sign_in/google_sign_in.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

// Set via --dart-define=GOOGLE_WEB_CLIENT_ID=YOUR_WEB_CLIENT_ID.apps.googleusercontent.com
const String kGoogleWebClientId =
    String.fromEnvironment('GOOGLE_WEB_CLIENT_ID', defaultValue: '');

final GoogleSignIn _google = GoogleSignIn(
  clientId: kIsWeb && kGoogleWebClientId.isNotEmpty ? kGoogleWebClientId : null,
  scopes: ['email', 'openid', 'profile'],
);

class GoogleAuthService {
  final GoogleSignIn _google = GoogleSignIn(scopes: ['email', 'openid', 'profile']);
  final FlutterSecureStorage _secure = const FlutterSecureStorage();

  Future<GoogleSignInAccount?> signIn() async {
    final acct = await _google.signIn();        // opens Google UI (iOS/Android/Web)
    if (acct == null) return null;              // user canceled
    final auth = await acct.authentication;

    await _secure.write(key: 'google_id_token', value: auth.idToken);
    await _secure.write(key: 'google_access_token', value: auth.accessToken);
    await _secure.write(key: 'profile_name', value: acct.displayName ?? '');
    await _secure.write(key: 'profile_email', value: acct.email);
    await _secure.write(key: 'profile_photo', value: acct.photoUrl ?? '');
    return acct;
  }

  Future<void> signOut({bool wipeLocal = false}) async {
    try { await _google.signOut(); } catch (_) {}
    if (wipeLocal) {
      await const FlutterSecureStorage().deleteAll();
    }
  }

  Future<String?> get idToken async => _secure.read(key: 'google_id_token');
  Future<String?> get accessToken async => _secure.read(key: 'google_access_token');
  Future<String?> get name async => _secure.read(key: 'profile_name');
  Future<String?> get email async => _secure.read(key: 'profile_email');
  Future<String?> get photo async => _secure.read(key: 'profile_photo');
}
