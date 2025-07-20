import 'dart:convert';
import 'dart:html' as html;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:intl/intl.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List? _imageBytes;
  String? _result;
  bool _loading = false;
  bool _darkMode = false;
  List<Map<String, String>> _history = [];

  Future<void> _pickImage() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result != null && result.files.single.bytes != null) {
      setState(() {
        _imageBytes = result.files.single.bytes!;
        _result = null;
      });
    }
  }

  Future<void> _captureFromCamera() async {
    final html.FileUploadInputElement uploadInput = html.FileUploadInputElement();
    uploadInput.accept = 'image/*';
    uploadInput.click();

    await uploadInput.onChange.first;

    if (uploadInput.files!.isNotEmpty) {
      final file = uploadInput.files!.first;
      final reader = html.FileReader();
      reader.readAsArrayBuffer(file);
      await reader.onLoad.first;
      setState(() {
        _imageBytes = reader.result as Uint8List;
        _result = null;
      });
    }
  }

  Future<void> _submitImage() async {
    if (_imageBytes == null) return;
    setState(() => _loading = true);

    try {
      final uri = Uri.parse(
          'https://turbo-spoon-7vxrgr7gjqxv3xp5r-8000.app.github.dev/predict/anemia');

      final request = http.MultipartRequest('POST', uri)
        ..files.add(http.MultipartFile.fromBytes(
          'file',
          _imageBytes!,
          filename: 'upload.jpg',
          contentType: MediaType('image', 'jpeg'),
        ));

      final response = await request.send();
      final respStr = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final json = jsonDecode(respStr);
        final resultText = json['anemic'] ? 'Anemic' : 'Not Anemic';

        final timestamp = DateFormat('yyyy-MM-dd HH:mm:ss').format(DateTime.now());
        final entry = {'timestamp': timestamp, 'result': resultText};

        setState(() {
          _result = resultText;
          _history.insert(0, entry);
        });

        html.window.localStorage['ishi_test_history'] = jsonEncode(_history);
      } else {
        setState(() => _result = 'Error: ${response.statusCode}');
      }
    } catch (e) {
      setState(() => _result = 'Network error: $e');
    } finally {
      setState(() => _loading = false);
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
    final theme = _darkMode ? ThemeData.dark() : ThemeData.light();
    return MaterialApp(
      theme: theme,
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          title: const Text('ISHI – Anemia Check (Web)'),
          actions: [
            IconButton(
              icon: Icon(_darkMode ? Icons.wb_sunny : Icons.nights_stay),
              onPressed: () => setState(() => _darkMode = !_darkMode),
              tooltip: 'Toggle Dark Mode',
            )
          ],
        ),
        body: SingleChildScrollView(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
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
                    const Text('Selected Image:',
                        style: TextStyle(fontWeight: FontWeight.bold)),
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
                      color: _result == 'Anemic'
                          ? Colors.red[100]
                          : Colors.green[100],
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Text(
                          _result!,
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: _result == 'Anemic'
                                ? Colors.red[800]
                                : Colors.green[800],
                          ),
                        ),
                      ),
                    ),
                  const SizedBox(height: 20),
                  Wrap(
                    spacing: 10,
                    runSpacing: 10,
                    alignment: WrapAlignment.center,
                    children: [
                      ElevatedButton(
                        onPressed: _pickImage,
                        child: const Text('Upload Eyelid Image'),
                      ),
                      ElevatedButton(
                        onPressed: kIsWeb ? _captureFromCamera : null,
                        child: const Text('Capture from Webcam'),
                      ),
                      ElevatedButton(
                        onPressed:
                            _imageBytes == null || _loading ? null : _submitImage,
                        child: const Text('Check for Anemia'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 30),
                  if (_history.isNotEmpty) ...[
                    const Divider(),
                    const Text('Test History',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 16)),
                    const SizedBox(height: 10),
                    ..._history
                        .map((entry) => ListTile(
                              leading: Icon(
                                entry['result'] == 'Anemic'
                                    ? Icons.warning
                                    : Icons.check_circle,
                                color: entry['result'] == 'Anemic'
                                    ? Colors.red
                                    : Colors.green,
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
      ),
    );
  }
}
