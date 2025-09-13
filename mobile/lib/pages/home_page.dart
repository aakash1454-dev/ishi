import 'dart:html' as html; // for reliable link opening on web
import 'package:flutter/material.dart';
import '../services/news_api.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late Future<List<NewsItem>> _news;

  @override
  void initState() {
    super.initState();
    _news = NewsApi.fetch();
  }

  Future<void> _refreshNews() async {
    setState(() => _news = NewsApi.fetch());
    await _news;
  }

  Widget _newsCard(NewsItem n) {
    return Card(
      elevation: 0,
      child: ListTile(
        leading: const Icon(Icons.article),
        title: Text(n.title),
        subtitle: Text(n.source),
        onTap: () {
          // Open in a new tab (most reliable in web/Codespaces)
          html.window.open(n.url, '_blank');
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: RefreshIndicator(
        onRefresh: _refreshNews,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 90), // space for floating nav
          children: [
            // Logo + title
            Column(
              children: [
                Image.asset('assets/logo.png', height: 100),
                const SizedBox(height: 12),
                const Text(
                  'Iron Strong Health Initiative, Inc',
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 24),
              ],
            ),

            // Health News only
            const Text('Health News', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            FutureBuilder<List<NewsItem>>(
              future: _news,
              builder: (context, snap) {
                if (snap.connectionState == ConnectionState.waiting) {
                  return const Padding(
                    padding: EdgeInsets.all(24.0),
                    child: Center(child: CircularProgressIndicator()),
                  );
                }
                if (snap.hasError) {
                  return Card(
                    color: Colors.red.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Text('News error: ${snap.error}', style: const TextStyle(color: Colors.red)),
                    ),
                  );
                }
                final items = snap.data ?? const <NewsItem>[];
                if (items.isEmpty) return const Text('No news available right now.');
                return Column(children: items.map(_newsCard).toList());
              },
            ),
          ],
        ),
      ),
    );
  }
}
