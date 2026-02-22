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
import 'package:image/image.dart' as img;

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
  final _imageContainerKey = GlobalKey();

  Uint8List? _imageBytes;
  String? _result;
  String? _detail;
  bool _loading = false;
  List<Map<String, String>> _history = [];
  bool _showInstructions = true; // Show guide before camera
  
  // For adjustable guide overlay
  Offset _guideOffset = const Offset(80, 120);
  double _guideScale = 1.0;
  final GlobalKey<DraggableCrescentGuideState> _crescentKey = GlobalKey();
  
  // For image zoom/pan
  final TransformationController _transformController = TransformationController();
  double _imageScale = 1.0;
  
  // Track if user is touching the image area (to disable page scroll)
  bool _touchingImageArea = false;
  
  // Container dimensions for crop calculation
  Size _containerSize = Size.zero;

  Uri? get _apiBaseUri {
    if (_apiBase.isEmpty) return null;
    final u = Uri.tryParse(_apiBase);
    return (u == null || !u.hasScheme) ? null : u;
  }

  Uri? _predictUri() => _apiBaseUri == null ? null : _joinPath(_apiBaseUri!, '/predict');
  Uri? _healthUri() => _apiBaseUri == null ? null : _joinPath(_apiBaseUri!, '/health');

  void _resetImageTransform() {
    _transformController.value = Matrix4.identity();
    _imageScale = 1.0;
  }

  Future<void> _pickFromGallery() async {
    final x = await _picker.pickImage(source: ImageSource.gallery, imageQuality: 95);
    if (x == null) return;
    final bytes = await x.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _result = null;
      _detail = null;
      // Reset guide position for new image
      _guideOffset = const Offset(80, 120);
      _guideScale = 1.0;
      _resetImageTransform();
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
      _guideOffset = const Offset(80, 120);
      _guideScale = 1.0;
      _resetImageTransform();
    });
  }

  /// Crop the image based on guide position and return cropped bytes
  /// Accounts for image zoom/pan via InteractiveViewer
  Future<Uint8List?> _cropImageToGuide() async {
    if (_imageBytes == null) {
      debugPrint('[CROP] No image bytes');
      return null;
    }
    
    try {
      // Decode the original image
      final decoded = img.decodeImage(_imageBytes!);
      if (decoded == null) {
        debugPrint('[CROP] Failed to decode image');
        return null;
      }
      
      final origW = decoded.width.toDouble();
      final origH = decoded.height.toDouble();
      debugPrint('[CROP] Original image: ${origW.toInt()}x${origH.toInt()}');
      
      // Get container size
      final containerBox = _imageContainerKey.currentContext?.findRenderObject() as RenderBox?;
      if (containerBox == null) {
        debugPrint('[CROP] Container not found, sending full image');
        return null;
      }
      final containerW = containerBox.size.width;
      final containerH = containerBox.size.height;
      debugPrint('[CROP] Container: ${containerW.toInt()}x${containerH.toInt()}');
      
      // Get transformation from InteractiveViewer
      final matrix = _transformController.value;
      final txTranslateX = matrix.getTranslation().x;
      final txTranslateY = matrix.getTranslation().y;
      final txScale = matrix.getMaxScaleOnAxis();
      debugPrint('[CROP] Transform: scale=$txScale, tx=$txTranslateX, ty=$txTranslateY');
      
      // Calculate base displayed image size (before zoom) using BoxFit.contain
      final containerAspect = containerW / containerH;
      final imageAspect = origW / origH;
      
      double baseDisplayedW, baseDisplayedH, baseOffsetX, baseOffsetY;
      if (imageAspect > containerAspect) {
        baseDisplayedW = containerW;
        baseDisplayedH = containerW / imageAspect;
        baseOffsetX = 0;
        baseOffsetY = (containerH - baseDisplayedH) / 2;
      } else {
        baseDisplayedH = containerH;
        baseDisplayedW = containerH * imageAspect;
        baseOffsetX = (containerW - baseDisplayedW) / 2;
        baseOffsetY = 0;
      }
      
      // After zoom, the displayed image size changes
      final zoomedDisplayW = baseDisplayedW * txScale;
      final zoomedDisplayH = baseDisplayedH * txScale;
      
      // The image's top-left corner in container coordinates after transform
      // InteractiveViewer centers the scaled content, so we need to account for that
      final zoomedOffsetX = baseOffsetX * txScale + txTranslateX + (containerW - zoomedDisplayW) / 2 * (txScale - 1) / txScale;
      final zoomedOffsetY = baseOffsetY * txScale + txTranslateY + (containerH - zoomedDisplayH) / 2 * (txScale - 1) / txScale;
      
      debugPrint('[CROP] Base displayed: ${baseDisplayedW.toInt()}x${baseDisplayedH.toInt()}');
      debugPrint('[CROP] Zoomed displayed: ${zoomedDisplayW.toInt()}x${zoomedDisplayH.toInt()}');
      
      // Guide dimensions in container coordinates (horizontal crescent shape)
      final guideW = 160 * _guideScale;
      final guideH = 90 * _guideScale;
      debugPrint('[CROP] Guide: ${guideW.toInt()}x${guideH.toInt()} at (${_guideOffset.dx.toInt()}, ${_guideOffset.dy.toInt()})');
      
      // Convert guide position from container coords to original image coords
      // First, find where the guide is relative to the zoomed image
      final guideInZoomedX = _guideOffset.dx - (txTranslateX + baseOffsetX);
      final guideInZoomedY = _guideOffset.dy - (txTranslateY + baseOffsetY);
      
      // Then scale back to original image coordinates
      final scaleToOrig = origW / (baseDisplayedW * txScale);
      
      // Add padding around the guide for better context (15% padding)
      const padFactor = 0.15;
      final padX = guideW * padFactor;
      final padY = guideH * padFactor;
      
      int cropX = ((guideInZoomedX - padX) * scaleToOrig).round().clamp(0, decoded.width - 1);
      int cropY = ((guideInZoomedY - padY) * scaleToOrig).round().clamp(0, decoded.height - 1);
      int cropW = ((guideW + 2 * padX) * scaleToOrig).round();
      int cropH = ((guideH + 2 * padY) * scaleToOrig).round();
      
      // Clamp to image bounds
      if (cropX + cropW > decoded.width) cropW = decoded.width - cropX;
      if (cropY + cropH > decoded.height) cropH = decoded.height - cropY;
      
      // Ensure minimum size
      cropW = cropW.clamp(50, decoded.width - cropX);
      cropH = cropH.clamp(50, decoded.height - cropY);
      
      debugPrint('[CROP] Crop rect: x=$cropX, y=$cropY, w=$cropW, h=$cropH');
      
      // Perform the crop
      final cropped = img.copyCrop(decoded, x: cropX, y: cropY, width: cropW, height: cropH);
      debugPrint('[CROP] Cropped to: ${cropped.width}x${cropped.height}');
      
      // Encode back to JPEG
      final result = Uint8List.fromList(img.encodeJpg(cropped, quality: 95));
      debugPrint('[CROP] Encoded: ${result.length} bytes');
      return result;
    } catch (e, stack) {
      debugPrint('[CROP] Error: $e');
      debugPrint('[CROP] Stack: $stack');
      return null;
    }
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
      // Crop the image based on guide position
      final croppedBytes = await _cropImageToGuide();
      final bytesToSend = croppedBytes ?? _imageBytes!;
      
      final req = http.MultipartRequest('POST', uri)
        ..files.add(http.MultipartFile.fromBytes(
          'image', // server expects "image"
          bytesToSend,
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
        final cropper = croppedBytes != null ? 'manual_crop' : (m['cropper'] ?? 'n/a').toString();

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
    } catch (e, stack) {
      debugPrint('[SUBMIT] Error: $e');
      debugPrint('[SUBMIT] Stack: $stack');
      setState(() {
        _result = 'Error';
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

  @override
  void dispose() {
    _transformController.dispose();
    super.dispose();
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
        // Disable scrolling when user is touching the image area
        physics: _touchingImageArea 
            ? const NeverScrollableScrollPhysics() 
            : const AlwaysScrollableScrollPhysics(),
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
                  const Text('Position Crescent Over Conjunctiva:', 
                    style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  const Text(
                    '2 fingers = zoom • Drag L/R/T/B to shape • Drag center to move',
                    style: TextStyle(fontSize: 11, color: Colors.grey),
                  ),
                  const SizedBox(height: 10),
                  
                  // Listener detects touch to disable page scroll
                  Listener(
                    onPointerDown: (_) => setState(() => _touchingImageArea = true),
                    onPointerUp: (_) => setState(() => _touchingImageArea = false),
                    onPointerCancel: (_) => setState(() => _touchingImageArea = false),
                    child: Container(
                      key: _imageContainerKey,
                      height: 400,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.teal.shade300, width: 2),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: Stack(
                          children: [
                            // Zoomable/pannable image
                            Positioned.fill(
                              child: InteractiveViewer(
                                transformationController: _transformController,
                                minScale: 1.0,
                                maxScale: 6.0,
                                panEnabled: true,
                                scaleEnabled: true,
                                onInteractionUpdate: (details) {
                                  setState(() {
                                    _imageScale = _transformController.value.getMaxScaleOnAxis();
                                  });
                                },
                                child: Image.memory(
                                  _imageBytes!,
                                  fit: BoxFit.contain,
                                ),
                              ),
                            ),
                            
                            // Draggable crescent guide with 4 control points
                            // Note: DraggableCrescentGuide adds 35px padding internally
                            Positioned(
                              left: _guideOffset.dx - 35, // Offset for internal padding
                              top: _guideOffset.dy - 35,
                              child: DraggableCrescentGuide(
                                key: _crescentKey, // Key for reset functionality
                                width: 160 * _guideScale,
                                height: 90 * _guideScale,
                                guideColor: Colors.lightGreenAccent,
                                strokeWidth: 3.0 * _guideScale,
                                onMove: (delta) {
                                  setState(() {
                                    _guideOffset += delta;
                                    _guideOffset = Offset(
                                      _guideOffset.dx.clamp(0, 250),
                                      _guideOffset.dy.clamp(0, 300),
                                    );
                                  });
                                },
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ), // End Listener
                  
                  // Zoom and size controls
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade200,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text('Zoom: ${(_imageScale * 100).toInt()}%',
                          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500)),
                      ),
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Text('Crescent:', style: TextStyle(fontSize: 12)),
                          IconButton(
                            icon: const Icon(Icons.remove, size: 20),
                            onPressed: () => setState(() => _guideScale = (_guideScale - 0.15).clamp(0.4, 2.5)),
                            padding: EdgeInsets.zero,
                            constraints: const BoxConstraints(minWidth: 36),
                          ),
                          Text('${(_guideScale * 100).toInt()}%', style: const TextStyle(fontSize: 12)),
                          IconButton(
                            icon: const Icon(Icons.add, size: 20),
                            onPressed: () => setState(() => _guideScale = (_guideScale + 0.15).clamp(0.4, 2.5)),
                            padding: EdgeInsets.zero,
                            constraints: const BoxConstraints(minWidth: 36),
                          ),
                        ],
                      ),
                    ],
                  ),
                  
                  // Reset button - resets position, size, shape, and zoom
                  TextButton.icon(
                    onPressed: () {
                      setState(() {
                        _guideOffset = const Offset(80, 120);
                        _guideScale = 1.0;
                        _resetImageTransform();
                      });
                      // Also reset the crescent shape to default
                      _crescentKey.currentState?.resetShape();
                    },
                    icon: const Icon(Icons.refresh, size: 16),
                    label: const Text('Reset All', style: TextStyle(fontSize: 12)),
                  ),
                  const SizedBox(height: 6),
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
