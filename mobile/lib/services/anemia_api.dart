// lib/services/anemia_api.dart
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

// Configurable API base for your FastAPI backend.
// Override at run-time with: --dart-define=API_BASE=https://<codespace>-8000.app.github.dev
const String kApiBase =
    String.fromEnvironment('API_BASE', defaultValue: 'http://localhost:8000');

class AnemiaApi {
  /// Uploads an image file and returns a PNG mask (bytes).
  /// When [overlay] is true the server should return an overlay instead of raw mask.
  static Future<Uint8List> uploadAndGetMask(
    String filePath, {
    bool overlay = false,
  }) async {
    final uri = Uri.parse(
      "$kApiBase/anemia/pseudomask${overlay ? '?return_overlay=true' : ''}",
    );

    final req = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath(
          'file', // keep as 'file' to match your pseudomask endpoint
          filePath,
          contentType: MediaType('image', 'jpeg'), // ok for jpg; server usually accepts png too
        ),
      );

    final streamed = await req.send();
    final resp = await http.Response.fromStream(streamed);

    if (resp.statusCode != 200) {
      throw Exception("Server ${resp.statusCode}: ${resp.body}");
    }
    return resp.bodyBytes; // PNG bytes
  }
}
