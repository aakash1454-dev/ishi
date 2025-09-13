// lib/services/news_api.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/api.dart';

class NewsItem {
  final String title, source, url;
  final String? image;
  NewsItem({required this.title, required this.source, required this.url, this.image});

  factory NewsItem.fromJson(Map<String, dynamic> j) => NewsItem(
    title: j['title'] ?? '',
    source: j['source'] ?? '',
    url: j['url'] ?? '',
    image: j['image'],
  );
}

class NewsApi {
  static Future<List<NewsItem>> fetch({int limit = 12}) async {
    final u = Uri.parse('${ApiConfig.newsUrl()}?limit=$limit');
    final r = await http.get(u);
    if (r.statusCode != 200) {
      throw Exception('News HTTP ${r.statusCode}: ${r.body}');
    }
    final List data = jsonDecode(r.body);
    return data.map((e) => NewsItem.fromJson(e)).toList();
  }
}
