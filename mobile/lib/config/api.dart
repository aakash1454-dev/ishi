// lib/config/api.dart
class ApiConfig {
  static const String baseUrl =
      'https://miniature-spoon-7qw5gg7qwr53xrjq-8000.app.github.dev';

  static String predictUrl() => '$baseUrl/predict';
  static String newsUrl() => '$baseUrl/news'; // used by Home news feed
}

