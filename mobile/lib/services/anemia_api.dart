import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import '../config/api.dart';

class AnemiaApi {
  static Future<Uint8List> uploadAndGetMask(String filePath, {bool overlay = false}) async {
    final uri = Uri.parse("$kApiBase/anemia/pseudomask${overlay ? '?return_overlay=true' : ''}");
    final req = http.MultipartRequest('POST', uri);
    req.files.add(await http.MultipartFile.fromPath(
      'file',
      filePath,
      contentType: MediaType('image', 'jpeg'), // ok for jpg/png
    ));
    final resp = await http.Response.fromStream(await req.send());
    if (resp.statusCode != 200) {
      throw Exception("Server ${resp.statusCode}: ${resp.body}");
    }
    return resp.bodyBytes; // PNG bytes
  }
}
