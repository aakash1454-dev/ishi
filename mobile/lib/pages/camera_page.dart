// lib/pages/camera_page.dart
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:intl/intl.dart';

// Import the new conjunctiva guide widget
import '../widgets/conjunctiva_guide.dart';

/// Read the backend base URL from a build-time define:
/// flutter run/build ... --dart-define=API_BASE_URL=https://ishi-api.onrender.com
const String _apiBase =
    String.fromEnvironment('API_BASE_URL', defaultValue: '');

Uri _joinPath(Uri base, String path) {
  if (path.startsWith('/')) {
    return base.replace(path: path, query: '');
  }
  final p = base.path.endsWith('/') ? '${base.path}$path' : '${base.path}/$path';
  return base.replace(path: p, query: '');
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  final _picker = ImagePicker();

  Uint8List? _imageBytes;
  String? _result;
  String? _detail;
  bool _loading = false;
  List<Map<String, String>> _history = [];
  bool _showInstructions = true; // Show guide before camera
  
  // For adjustable guide overlay
  Offset _guideOffset = Offset.zero;
  double _guideScale = 1.0;
  double _baseScale = 1.0; // For tracking scale at gesture start

  Uri? get _apiBaseUri {
    if (_apiBase.isEmpty) return null;
    final u = Uri.tryParse(_apiBase);
    return (u == null || !u.hasScheme) ? null : u;
  }

  Uri? _predictUri() => _apiBaseUri == null ? null : _joinPath(_apiBaseUri!, '/predict');
  Uri? _healthUri() => _apiBaseUri == null ? null : _joinPath(_apiBaseUri!, '/health');

  Future<void> _pickFromGallery() async {
    final x = await _picker.pickImage(source: ImageSource.gallery, imageQuality: 95);
    if (x == null) return;
    final bytes = await x.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _result = null;
      _detail = null;
      // Reset guide position for new image
      _guideOffset = const Offset(100, 100);
      _guideScale = 1.0;
    });
  }

  Future<void> _takePhoto() async {
    // Works on Android/iOS; on web it falls back to file input.
    final x = await _picker.pickImage(source: ImageSource.camera, imageQuality: 95);
    if (x == null) return;
    final bytes = await x.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _result = null;
      _detail = null;
      // Reset guide position for new image
      _guideOffset = const Offset(100, 100);
      _guideScale = 1.0;
    });
  }

  Future<void> _submitImage() async {
    if (_imageBytes == null) return;
    final uri = _predictUri();
    if (uri == null) {
      setState(() {
        _result = 'API not configured';
        _detail = 'Pass --dart-define=API_BASE_URL=https://your-api';
      });
      return;
    }

    setState(() => _loading = true);

    try {
      final req = http.MultipartRequest('POST', uri)
        ..files.add(http.MultipartFile.fromBytes(
          'image', // server expects "image"
          _imageBytes!,
          filename: 'upload.jpg',
          contentType: MediaType('image', 'jpeg'),
        ));

      final res = await req.send();
      final body = await res.stream.bytesToString();

      if (res.statusCode == 200) {
        final m = jsonDecode(body) as Map<String, dynamic>;
        final isAnemic = m['anemic'] == true;
        final score = (m['score'] is num) ? (m['score'] as num).toDouble() : 0.0;
        final pct = (score * 100).toStringAsFixed(1);
        final cropper = (m['cropper'] ?? 'n/a').toString();

        final resultText = isAnemic ? 'Anemic' : 'Not Anemic';
        final detailText = 'Score: $pct% • Cropper: $cropper';

        final ts = DateFormat('yyyy-MM-dd HH:mm:ss').format(DateTime.now());
        final entry = {'timestamp': ts, 'result': resultText};

        // Persist history
        final prefs = await SharedPreferences.getInstance();
        setState(() {
          _result = resultText;
          _detail = detailText;
          _history.insert(0, entry);
        });
        await prefs.setString('ishi_test_history', jsonEncode(_history));
      } else {
        setState(() {
          _result = 'Error ${res.statusCode}';
          _detail = body;
        });
      }
    } catch (e) {
      setState(() {
        _result = 'Network error';
        _detail = e.toString();
      });
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _testHealth() async {
    final u = _healthUri();
    if (u == null) {
      setState(() {
        _result = 'API not configured';
        _detail = 'Pass --dart-define=API_BASE_URL=https://your-api';
      });
      return;
    }
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

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString('ishi_test_history');
    if (saved != null) {
      final decoded = jsonDecode(saved);
      final list = (decoded as List)
          .map((e) => Map<String, String>.from(e as Map))
          .toList();
      setState(() {
        _history = list;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  void _dismissInstructions() {
    setState(() {
      _showInstructions = false;
    });
  }

  void _showInstructionsAgain() {
    setState(() {
      _showInstructions = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    final api = _apiBaseUri;
    final apiText = api == null
        ? 'API: (not set)'
        : 'API: ${api.scheme}://${api.host}${api.hasPort ? ':${api.port}' : ''}${api.path}';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Anemia Checker'),
        actions: [
          // Help button to show instructions again
          IconButton(
            icon: const Icon(Icons.help_outline),
            onPressed: _showInstructionsAgain,
            tooltip: 'How to capture',
          ),
        ],
      ),
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
                    Flexible(
                      child: SelectableText(
                        apiText,
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),

                // Show instruction card if first time or help requested
                if (_showInstructions && _imageBytes == null) ...[
                  CaptureInstructionCard(
                    onContinue: _dismissInstructions,
                  ),
                  const SizedBox(height: 16),
                ] else ...[
                  // Regular content when not showing instructions
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
                  const SizedBox(height: 16),
                  
                  // Quick visual reminder of the guide
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.teal.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.teal.withOpacity(0.3)),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.info_outline, color: Colors.teal, size: 20),
                        SizedBox(width: 8),
                        Text(
                          'Position your lower eyelid in the crescent guide',
                          style: TextStyle(color: Colors.teal, fontSize: 13),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                ],

                if (_imageBytes != null) ...[
                  const Text('Adjust Image & Position Guide:', 
                    style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  const Text(
                    'Pinch to zoom image • Drag green guide to align',
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                  const SizedBox(height: 10),
                  
                  // Interactive image with draggable guide
                  Container(
                    height: 300,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Stack(
                        children: [
                          // Zoomable/pannable image
                          InteractiveViewer(
                            minScale: 0.5,
                            maxScale: 4.0,
                            child: Image.memory(
                              _imageBytes!,
                              fit: BoxFit.contain,
                            ),
                          ),
                          
                          // Draggable & scalable guide overlay
                          Positioned(
                            left: _guideOffset.dx,
                            top: _guideOffset.dy,
                            child: GestureDetector(
                              onScaleStart: (_) {
                                _baseScale = _guideScale;
                              },
                              onScaleUpdate: (details) {
                                setState(() {
                                  // Handle drag (focal point delta)
                                  _guideOffset += details.focalPointDelta;
                                  // Handle pinch scale
                                  _guideScale = (_baseScale * details.scale).clamp(0.3, 2.0);
                                });
                              },
                              child: Opacity(
                                opacity: 0.7,
                                child: ConjunctivaGuide(
                                  width: 120 * _guideScale,
                                  height: 50 * _guideScale,
                                  guideColor: Colors.green,
                                  showInstructions: false,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  
                  // Guide size controls
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text('Guide size:', style: TextStyle(fontSize: 12)),
                      IconButton(
                        icon: const Icon(Icons.remove_circle_outline, size: 20),
                        onPressed: () => setState(() {
                          _guideScale = (_guideScale - 0.1).clamp(0.3, 2.0);
                        }),
                      ),
                      Text('${(_guideScale * 100).toInt()}%', 
                        style: const TextStyle(fontSize: 12)),
                      IconButton(
                        icon: const Icon(Icons.add_circle_outline, size: 20),
                        onPressed: () => setState(() {
                          _guideScale = (_guideScale + 0.1).clamp(0.3, 2.0);
                        }),
                      ),
                      const SizedBox(width: 8),
                      TextButton(
                        onPressed: () => setState(() {
                          _guideOffset = Offset.zero;
                          _guideScale = 1.0;
                        }),
                        child: const Text('Reset', style: TextStyle(fontSize: 12)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
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

                // Only show capture buttons after dismissing instructions
                if (!_showInstructions || _imageBytes != null)
                  Wrap(
                    spacing: 10,
                    runSpacing: 10,
                    alignment: WrapAlignment.center,
                    children: [
                      ElevatedButton.icon(
                        onPressed: _pickFromGallery,
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Upload Image'),
                      ),
                      ElevatedButton.icon(
                        onPressed: _takePhoto,
                        icon: const Icon(Icons.camera_alt),
                        label: Text(kIsWeb ? 'Capture' : 'Take Photo'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal,
                          foregroundColor: Colors.white,
                        ),
                      ),
                      ElevatedButton.icon(
                        onPressed: _imageBytes == null || _loading ? null : _submitImage,
                        icon: const Icon(Icons.science),
                        label: const Text('Analyze'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.deepPurple,
                          foregroundColor: Colors.white,
                        ),
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
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
