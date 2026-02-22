// lib/widgets/conjunctiva_guide.dart
import 'package:flutter/material.dart';
import 'dart:math' as math;

/// A horizontal crescent guide (like a smile) for positioning the conjunctiva.
/// Has 4 draggable control points that can be adjusted by touch.
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
          // Simple static crescent for instruction display
          CustomPaint(
            size: Size(width, height),
            painter: _SimpleCrescentPainter(
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

/// Interactive crescent with 4 draggable control points
class DraggableCrescentGuide extends StatefulWidget {
  final double width;
  final double height;
  final Color guideColor;
  final double strokeWidth;
  
  // Callback when user drags the center to move the whole crescent
  final Function(Offset delta)? onMove;
  
  // Callback when shape changes
  final Function(Offset left, Offset right, Offset top, Offset bottom)? onShapeChanged;

  const DraggableCrescentGuide({
    super.key,
    this.width = 150,
    this.height = 80,
    this.guideColor = Colors.lightGreenAccent,
    this.strokeWidth = 3.0,
    this.onMove,
    this.onShapeChanged,
  });
  
  @override
  State<DraggableCrescentGuide> createState() => DraggableCrescentGuideState();
}

// Made public so it can be accessed via GlobalKey for reset
class DraggableCrescentGuideState extends State<DraggableCrescentGuide> {
  // Padding added to widget to ensure touch areas stay inside hit test bounds
  static const double _padding = 35.0;
  
  /// Call this to reset the crescent to its default shape
  void resetShape() {
    setState(() {
      _initializeRatios();
    });
  }
  
  // Control points stored as RATIOS (0-1) of the INNER area (excluding padding)
  late double _leftX, _leftY;
  late double _rightX, _rightY;
  late double _topX, _topY;
  late double _bottomX, _bottomY;
  
  // Track which point is being dragged (null = none, -1 = center/move)
  int? _draggingPoint;
  
  // Inner area dimensions (widget size minus padding)
  double get _innerWidth => widget.width;
  double get _innerHeight => widget.height;
  
  // Convert ratio to actual position (with padding offset)
  Offset _getPosition(int index) {
    double x, y;
    switch (index) {
      case 0: x = _leftX; y = _leftY; break;
      case 1: x = _rightX; y = _rightY; break;
      case 2: x = _topX; y = _topY; break;
      case 3: x = _bottomX; y = _bottomY; break;
      default: return Offset.zero;
    }
    return Offset(
      _padding + x * _innerWidth,
      _padding + y * _innerHeight,
    );
  }
  
  @override
  void initState() {
    super.initState();
    _initializeRatios();
  }
  
  void _initializeRatios() {
    // Default crescent shape as ratios (0-1)
    _leftX = 0.0; _leftY = 0.5;
    _rightX = 1.0; _rightY = 0.5;
    _topX = 0.5; _topY = 0.2;
    _bottomX = 0.5; _bottomY = 0.9;
  }
  
  // Update a point from screen position (accounting for padding)
  void _updatePointFromScreenPos(int index, Offset screenPos) {
    // Convert screen position to ratio (subtract padding, divide by inner size)
    final rawX = (screenPos.dx - _padding) / _innerWidth;
    final rawY = (screenPos.dy - _padding) / _innerHeight;
    
    // Keep points within reasonable bounds (staying inside the widget visually)
    // Allow movement within the padded area (-0.2 to 1.2 means 20% outside inner area,
    // but still inside the total widget bounds with 35px padding)
    double x, y;
    if (index == 0 || index == 1) {
      // L and R: tighter X limits, moderate Y
      x = rawX.clamp(-0.1, 1.1);
      y = rawY.clamp(-0.3, 1.3);
    } else {
      // T and B: moderate X, wider Y but stay in bounds
      x = rawX.clamp(-0.2, 1.2);
      y = rawY.clamp(-0.35, 1.35); // Stay within padded bounds
    }
    
    setState(() {
      switch (index) {
        case 0: _leftX = x; _leftY = y; break;
        case 1: _rightX = x; _rightY = y; break;
        case 2: _topX = x; _topY = y; break;
        case 3: _bottomX = x; _bottomY = y; break;
      }
    });
  }
  
  void _notifyChange() {
    // Report positions without padding for external use
    widget.onShapeChanged?.call(
      Offset(_leftX * _innerWidth, _leftY * _innerHeight),
      Offset(_rightX * _innerWidth, _rightY * _innerHeight),
      Offset(_topX * _innerWidth, _topY * _innerHeight),
      Offset(_bottomX * _innerWidth, _bottomY * _innerHeight),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Total size includes padding for touch areas
    final totalWidth = widget.width + _padding * 2;
    final totalHeight = widget.height + _padding * 2;
    
    // Get current positions (with padding)
    final left = _getPosition(0);
    final right = _getPosition(1);
    final top = _getPosition(2);
    final bottom = _getPosition(3);
    
    return SizedBox(
      width: totalWidth,
      height: totalHeight,
      child: Stack(
        children: [
          // The crescent shape (offset by padding)
          Positioned(
            left: _padding,
            top: _padding,
            width: _innerWidth,
            height: _innerHeight,
            child: GestureDetector(
              behavior: HitTestBehavior.translucent,
              onPanStart: (_) => _draggingPoint = -1,
              onPanUpdate: (details) {
                if (_draggingPoint == -1) {
                  widget.onMove?.call(details.delta);
                }
              },
              onPanEnd: (_) => _draggingPoint = null,
              child: CustomPaint(
                size: Size(_innerWidth, _innerHeight),
                painter: _DraggableCrescentPainter(
                  color: widget.guideColor,
                  strokeWidth: widget.strokeWidth,
                  leftPoint: Offset(_leftX * _innerWidth, _leftY * _innerHeight),
                  rightPoint: Offset(_rightX * _innerWidth, _rightY * _innerHeight),
                  topControl: Offset(_topX * _innerWidth, _topY * _innerHeight),
                  bottomControl: Offset(_bottomX * _innerWidth, _bottomY * _innerHeight),
                ),
              ),
            ),
          ),
          
          // Control points - positioned within the padded area so hit testing works
          _buildControlPoint(0, left, 'L'),
          _buildControlPoint(1, right, 'R'),
          _buildControlPoint(2, top, 'T'),
          _buildControlPoint(3, bottom, 'B'),
        ],
      ),
    );
  }

  Widget _buildControlPoint(int index, Offset position, String label) {
    const double visualSize = 26;
    const double touchSize = 56;
    
    return Positioned(
      left: position.dx - touchSize / 2,
      top: position.dy - touchSize / 2,
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onPanStart: (_) {
          _draggingPoint = index;
        },
        onPanUpdate: (details) {
          if (_draggingPoint != index) return;
          // Get current position and add delta
          final currentPos = _getPosition(index);
          _updatePointFromScreenPos(index, currentPos + details.delta);
          _notifyChange();
        },
        onPanEnd: (_) => _draggingPoint = null,
        onPanCancel: () => _draggingPoint = null,
        child: Container(
          width: touchSize,
          height: touchSize,
          color: Colors.transparent,
          alignment: Alignment.center,
          child: Container(
            width: visualSize,
            height: visualSize,
            decoration: BoxDecoration(
              color: Colors.white,
              shape: BoxShape.circle,
              border: Border.all(
                color: _draggingPoint == index ? Colors.yellow : widget.guideColor,
                width: 3,
              ),
              boxShadow: [
                BoxShadow(
                  color: _draggingPoint == index 
                      ? Colors.yellow.withOpacity(0.8)
                      : Colors.black.withOpacity(0.5),
                  blurRadius: _draggingPoint == index ? 12 : 6,
                  spreadRadius: _draggingPoint == index ? 4 : 2,
                ),
              ],
            ),
            child: Center(
              child: Text(
                label,
                style: TextStyle(
                  color: _draggingPoint == index ? Colors.black : widget.guideColor,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Simple static crescent painter (horizontal smile shape)
class _SimpleCrescentPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;

  _SimpleCrescentPainter({
    required this.color,
    required this.strokeWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withOpacity(0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()
      ..color = color.withOpacity(0.15)
      ..style = PaintingStyle.fill;

    // Horizontal crescent (like a smile)
    final path = Path();
    
    final leftPoint = Offset(0, size.height * 0.4);
    final rightPoint = Offset(size.width, size.height * 0.4);
    final topControl = Offset(size.width * 0.5, size.height * 0.1);
    final bottomControl = Offset(size.width * 0.5, size.height * 0.9);
    
    // Start at left tip
    path.moveTo(leftPoint.dx, leftPoint.dy);
    
    // Top curve (inner, flatter)
    path.quadraticBezierTo(topControl.dx, topControl.dy, rightPoint.dx, rightPoint.dy);
    
    // Bottom curve (outer, more curved)
    path.quadraticBezierTo(bottomControl.dx, bottomControl.dy, leftPoint.dx, leftPoint.dy);
    
    path.close();

    canvas.drawPath(path, fillPaint);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// Crescent painter with 4 custom control points
class _DraggableCrescentPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;
  final Offset leftPoint;
  final Offset rightPoint;
  final Offset topControl;
  final Offset bottomControl;

  _DraggableCrescentPainter({
    required this.color,
    required this.strokeWidth,
    required this.leftPoint,
    required this.rightPoint,
    required this.topControl,
    required this.bottomControl,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withOpacity(0.95)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()
      ..color = color.withOpacity(0.2)
      ..style = PaintingStyle.fill;

    // Draw crescent using the 4 control points
    final path = Path();
    
    // Start at left tip
    path.moveTo(leftPoint.dx, leftPoint.dy);
    
    // Top curve (inner) - from left to right via top control
    path.quadraticBezierTo(topControl.dx, topControl.dy, rightPoint.dx, rightPoint.dy);
    
    // Bottom curve (outer) - from right back to left via bottom control
    path.quadraticBezierTo(bottomControl.dx, bottomControl.dy, leftPoint.dx, leftPoint.dy);
    
    path.close();

    // Draw fill then stroke
    canvas.drawPath(path, fillPaint);
    canvas.drawPath(path, paint);
    
    // Draw small dots at the tips
    final tipPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    canvas.drawCircle(leftPoint, strokeWidth * 1.5, tipPaint);
    canvas.drawCircle(rightPoint, strokeWidth * 1.5, tipPaint);
  }

  @override
  bool shouldRepaint(covariant _DraggableCrescentPainter oldDelegate) {
    return oldDelegate.leftPoint != leftPoint ||
           oldDelegate.rightPoint != rightPoint ||
           oldDelegate.topControl != topControl ||
           oldDelegate.bottomControl != bottomControl;
  }
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

