# Guide Example Image

To add a real example image showing proper eye positioning:

1. Take a clear photo showing:
   - Lower eyelid pulled down
   - Pink conjunctiva clearly visible
   - Good lighting

2. Save as `guide_example.png` in this folder

3. Add to pubspec.yaml:
   ```yaml
   flutter:
     assets:
       - assets/logo.png
       - assets/guide_example.png
   ```

4. Use in the CaptureInstructionCard:
   ```dart
   Image.asset('assets/guide_example.png', height: 150)
   ```

For now, the app uses the crescent guide widget as a visual reference.

