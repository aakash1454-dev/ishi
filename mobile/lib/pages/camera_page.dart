import 'dart:convert';
import 'dart:html' as html;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:intl/intl.dart';

/// Optional override at run-time:
/// flutter run -d web-server --web-hostname 0.0.0.0 --web-port 5287 \
///   --dart-define=ISHI_API=https://<your-forwarded-8000>.github.dev
const String _apiOverride =
    String.fromEnvironment('ISHI_API', defaultValue: '');

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  Uint8List? _imageBytes;
  String? _result;
  String? _detail;
  bool _loading = false;
  List<Map<String, String>> _history = [];

  /// Build the correct API base (always targets `/predict`)
  Uri _buildApiUri() {
    if (_apiOverride.isNotEmpty) {
      final base = Uri.parse(_apiOverride);
      return base.replace(path: '/predict', query: '');
    }
    final u = Uri.base; // where the Flutter web app is hosted
    if (u.host.endsWith('.github.dev')) {
      // Handle both "5287-xxxxx.github.dev" and "...-5287.app.github.dev"
      var host = u.host;
      host = host.replaceFirst(RegExp(r'^\d+-'), '8000-');      // 5287-… → 8000-…
      host = host.replaceFirst(RegExp(r'-\d+(?=\.)'), '-8000'); // …-5287. → …-8000.
      return Uri(scheme: u.scheme, host: host, path: '/predict');
    }
    // Local dev fallback
    return Uri(scheme: 'http', host: 'localhost', port: 8000, path: '/predict');
  }

  Future<void> _pickImage() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result != null && result.files.single.bytes != null) {
      setState(() {
        _imageBytes = result.files.single.bytes!;
        _result = null;
        _detail = null;
      });
    }
  }

  Future<void> _captureFromCamera() async {
    final uploadInput = html.FileUploadInputElement();
    uploadInput.accept = 'image/*';
    uploadInput.setAttribute('capture', 'environment'); // hint: rear camera
    uploadInput.click();

    await uploadInput.onChange.first;

    if (uploadInput.files?.isNotEmpty == true) {
      final file = uploadInput.files!.first;
      final reader = html.FileReader()..readAsArrayBuffer(file);
      await reader.onLoad.first;
      setState(() {
        _imageBytes = reader.result as Uint8List;
        _result = null;
        _detail = null;
      });
    }
  }

  Future<void> _submitImage() async {
    if (_imageBytes == null) return;
    setState(() => _loading = true);

    if (!kIsWeb) {
      setState(() {
        _result = 'Backend call skipped (non-web platform)';
        _loading = false;
      });
      return;
    }

    try {
      final uri = _buildApiUri();
      debugPrint('POST -> $uri'); // should end with /predict

      final req = http.MultipartRequest('POST', uri)
        ..files.add(http.MultipartFile.fromBytes(
          'image', // <-- MUST be "image"
          _imageBytes!,
          filename: 'upload.jpg',
          contentType: MediaType('image', 'jpeg'),
        ));

      final res = await req.send();
      final body = await res.stream.bytesToString();

      if (res.statusCode == 200) {
        final m = jsonDecode(body) as Map<String, dynamic>;
        final isAnemic = m['anemic'] == true;
        final score = (m['score'] ?? 0.0) * 1.0;
        final pct = (score is num) ? (score * 100).toStringAsFixed(1) : score.toString();
        final cropper = (m['cropper'] ?? 'n/a').toString();

        final resultText = isAnemic ? 'Anemic' : 'Not Anemic';
        final detailText = 'Score: $pct% • Cropper: $cropper';

        final timestamp = DateFormat('yyyy-MM-dd HH:mm:ss').format(DateTime.now());
        final entry = {'timestamp': timestamp, 'result': resultText};

        setState(() {
          _result = resultText;
          _detail = detailText;
          _history.insert(0, entry);
        });
        html.window.localStorage['ishi_test_history'] = jsonEncode(_history);
      } else {
        setState(() => _result = 'Error: ${res.statusCode}\n$body');
      }
    } catch (e) {
      setState(() => _result = 'Network error: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _testHealth() async {
    final base = _buildApiUri();
    final u = base.replace(path: '/health', query: '');
    debugPrint('GET  -> $u');
    try {
      final r = await http.get(u);
      setState(() {
        _result = 'Health ${r.statusCode}';
        _detail = r.body;
      });
    } catch (e) {
      setState(() {
        _result = 'Health error';
        _detail = e.toString();
      });
    }
  }

  @override
  void initState() {
    super.initState();
    final saved = html.window.localStorage['ishi_test_history'];
    if (saved != null) {
      final decoded = jsonDecode(saved);
      _history = List<Map<String, String>>.from(
        (decoded as List).map((e) => Map<String, String>.from(e)),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final api = _buildApiUri();
    final apiText =
        'API: ${api.scheme}://${api.host}${api.hasPort ? ':${api.port}' : ''}${api.path}';
    return Scaffold(
      appBar: AppBar(title: const Text('Anemia Checker')),
      body: SingleChildScrollView(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                // Connectivity row
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    OutlinedButton(onPressed: _testHealth, child: const Text('Test API')),
                    const SizedBox(width: 12),
                    Flexible(child: SelectableText(apiText, style: const TextStyle(fontSize: 12))),
                  ],
                ),
                const SizedBox(height: 16),

                const Text(
                  'About Anemia',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
                ),
                const SizedBox(height: 8),
                const Card(
                  child: Padding(
                    padding: EdgeInsets.all(12),
                    child: Text(
                      'Anemia is a condition where you lack enough healthy red blood cells to carry adequate oxygen to your body\'s tissues. Detecting it early can help prevent fatigue, weakness, and more serious complications.',
                      style: TextStyle(fontSize: 14),
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                if (_imageBytes != null) ...[
                  const Text('Selected Image:', style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.memory(_imageBytes!, height: 200),
                  ),
                  const SizedBox(height: 20),
                ],

                if (_loading)
                  const CircularProgressIndicator()
                else if (_result != null)
                  Card(
                    color: _result == 'Anemic' ? Colors.red[100] : Colors.green[100],
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          Text(
                            _result!,
                            style: TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: _result == 'Anemic' ? Colors.red[800] : Colors.green[800],
                            ),
                          ),
                          if (_detail != null) ...[
                            const SizedBox(height: 6),
                            Text(_detail!, style: const TextStyle(fontSize: 14)),
                          ]
                        ],
                      ),
                    ),
                  ),

                const SizedBox(height: 20),

                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  alignment: WrapAlignment.center,
                  children: [
                    ElevatedButton(onPressed: _pickImage, child: const Text('Upload Eyelid Image')),
                    ElevatedButton(
                      onPressed: kIsWeb ? _captureFromCamera : null,
                      child: const Text('Capture from Webcam'),
                    ),
                    ElevatedButton(
                      onPressed: _imageBytes == null || _loading ? null : _submitImage,
                      child: const Text('Check for Anemia'),
                    ),
                  ],
                ),

                const SizedBox(height: 30),

                if (_history.isNotEmpty) ...[
                  const Divider(),
                  const Text('Test History',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                  const SizedBox(height: 10),
                  ..._history
                      .map((entry) => ListTile(
                            leading: Icon(
                              entry['result'] == 'Anemic' ? Icons.warning : Icons.check_circle,
                              color: entry['result'] == 'Anemic' ? Colors.red : Colors.green,
                            ),
                            title: Text(entry['result']!),
                            subtitle: Text(entry['timestamp']!),
                          ))
                      .toList(),
                ]
              ],
            ),
          ),
        ),
      ),
    );
  }
}
