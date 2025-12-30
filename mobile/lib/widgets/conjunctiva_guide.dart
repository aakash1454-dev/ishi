// lib/widgets/conjunctiva_guide.dart
import 'package:flutter/material.dart';
import 'dart:math' as math;

/// A crescent-shaped overlay guide for positioning the conjunctiva.
/// Used both as a static instruction and as a camera overlay.
class ConjunctivaGuide extends StatelessWidget {
  final double width;
  final double height;
  final Color guideColor;
  final double strokeWidth;
  final bool showInstructions;
  final bool animated;

  const ConjunctivaGuide({
    super.key,
    this.width = 280,
    this.height = 120,
    this.guideColor = Colors.green,
    this.strokeWidth = 3.0,
    this.showInstructions = true,
    this.animated = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      height: height + (showInstructions ? 60 : 0),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // The crescent guide
          CustomPaint(
            size: Size(width, height),
            painter: _CrescentPainter(
              color: guideColor,
              strokeWidth: strokeWidth,
            ),
          ),
          if (showInstructions) ...[
            const SizedBox(height: 12),
            Text(
              'Align your lower eyelid here',
              style: TextStyle(
                color: guideColor,
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ],
      ),
    );
  }
}

/// Custom painter for the crescent shape
class _CrescentPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;

  _CrescentPainter({
    required this.color,
    required this.strokeWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withOpacity(0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()
      ..color = color.withOpacity(0.1)
      ..style = PaintingStyle.fill;

    // Create crescent path
    final path = Path();
    
    // Outer arc (bottom of crescent)
    final outerRect = Rect.fromLTWH(
      size.width * 0.05,
      size.height * 0.1,
      size.width * 0.9,
      size.height * 1.2,
    );
    path.addArc(outerRect, math.pi * 0.15, math.pi * 0.7);
    
    // Inner arc (top of crescent) - creates the crescent shape
    final innerRect = Rect.fromLTWH(
      size.width * 0.1,
      size.height * 0.25,
      size.width * 0.8,
      size.height * 0.9,
    );
    path.arcTo(innerRect, math.pi * 0.85, -math.pi * 0.7, false);
    
    path.close();

    // Draw fill first, then stroke
    canvas.drawPath(path, fillPaint);
    canvas.drawPath(path, paint);

    // Draw corner markers for alignment
    _drawCornerMarkers(canvas, size, paint);
  }

  void _drawCornerMarkers(Canvas canvas, Size size, Paint paint) {
    final markerPaint = Paint()
      ..color = color.withOpacity(0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    const markerSize = 15.0;
    
    // Top-left corner
    canvas.drawLine(
      Offset(0, markerSize),
      const Offset(0, 0),
      markerPaint,
    );
    canvas.drawLine(
      const Offset(0, 0),
      Offset(markerSize, 0),
      markerPaint,
    );
    
    // Top-right corner
    canvas.drawLine(
      Offset(size.width - markerSize, 0),
      Offset(size.width, 0),
      markerPaint,
    );
    canvas.drawLine(
      Offset(size.width, 0),
      Offset(size.width, markerSize),
      markerPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// A full-screen camera overlay with the crescent guide
class CameraOverlay extends StatelessWidget {
  final bool isReady;
  final VoidCallback? onCapture;

  const CameraOverlay({
    super.key,
    this.isReady = false,
    this.onCapture,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Semi-transparent background
        Container(
          color: Colors.black.withOpacity(0.3),
        ),
        
        // Center the guide
        Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Instructions at top
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text(
                  'Pull down your lower eyelid\nand position the pink area in the guide',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              
              const SizedBox(height: 40),
              
              // The crescent guide
              ConjunctivaGuide(
                width: 300,
                height: 130,
                guideColor: isReady ? Colors.green : Colors.white,
                strokeWidth: 4,
                showInstructions: false,
              ),
              
              const SizedBox(height: 20),
              
              // Ready indicator
              AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: isReady ? Colors.green : Colors.orange,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  isReady ? '✓ Ready to capture' : 'Align your eye...',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
        
        // Capture button at bottom
        Positioned(
          bottom: 40,
          left: 0,
          right: 0,
          child: Center(
            child: GestureDetector(
              onTap: onCapture,
              child: Container(
                width: 70,
                height: 70,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white,
                  border: Border.all(
                    color: isReady ? Colors.green : Colors.grey,
                    width: 4,
                  ),
                ),
                child: Icon(
                  Icons.camera_alt,
                  size: 35,
                  color: isReady ? Colors.green : Colors.grey,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// Instruction card shown before camera capture
class CaptureInstructionCard extends StatelessWidget {
  final VoidCallback onContinue;

  const CaptureInstructionCard({
    super.key,
    required this.onContinue,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'How to Capture',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            
            // Visual guide
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(12),
              ),
              child: const ConjunctivaGuide(
                width: 125,
                height: 50,
                guideColor: Colors.teal,
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Steps
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _InstructionStep(
                  number: 1,
                  text: 'Find good lighting (natural light is best)',
                ),
                _InstructionStep(
                  number: 2,
                  text: 'Gently pull down your lower eyelid',
                ),
                _InstructionStep(
                  number: 3,
                  text: 'Position the pink inner eyelid in the crescent guide',
                ),
                _InstructionStep(
                  number: 4,
                  text: 'Hold steady and capture',
                ),
              ],
            ),
            
            const SizedBox(height: 20),
            
            ElevatedButton.icon(
              onPressed: onContinue,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Got it, Open Camera'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _InstructionStep extends StatelessWidget {
  final int number;
  final String text;

  const _InstructionStep({
    required this.number,
    required this.text,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(
              color: Colors.teal,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Center(
              child: Text(
                '$number',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(fontSize: 14),
            ),
          ),
        ],
      ),
    );
  }
}

