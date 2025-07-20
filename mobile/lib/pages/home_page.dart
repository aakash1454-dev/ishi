import 'camera_page.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String name = 'Guest';
  String language = 'English';

  final List<Map<String, String>> events = [
    {'title': 'Anemia Camp – Aug 5', 'date': '2025-08-05'},
    {'title': 'Free Screening – Sep 10', 'date': '2025-09-10'},
  ];

  final List<Map<String, String>> blogs = [
    {'title': 'ISHI at Rutgers', 'summary': 'Our outreach team visited...'},
    {'title': 'Summer Recap', 'summary': 'We helped 1200 patients...'},
  ];

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  Future<void> _loadProfile() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      name = prefs.getString('name') ?? 'Guest';
      language = prefs.getString('language') ?? 'English';
    });
  }

  Widget _quickActionTile(String label, IconData icon, VoidCallback onTap) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(padding: const EdgeInsets.all(12)),
      onPressed: onTap,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 30),
          const SizedBox(height: 10),
          Text(label, textAlign: TextAlign.center),
        ],
      ),
    );
  }

  Widget _eventCard(String title, String date) {
    return Card(
      child: ListTile(
        leading: const Icon(Icons.event),
        title: Text(title),
        subtitle: Text(date),
      ),
    );
  }

  Widget _newsCard(String title, String summary) {
    return Card(
      child: ListTile(
        leading: const Icon(Icons.article),
        title: Text(title),
        subtitle: Text(summary),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () async {
          const url = 'https://www.yourdonationpage.com';
          if (await canLaunchUrl(Uri.parse(url))) {
            await launchUrl(Uri.parse(url));
          }
        },
        icon: const Icon(Icons.favorite),
        label: const Text('Donate'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
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
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Hello, $name!',
                style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              ),
              const Icon(Icons.settings),
            ],
          ),
          const SizedBox(height: 20),

          const Text('Quick Actions', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
          const SizedBox(height: 10),
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            crossAxisSpacing: 12,
            mainAxisSpacing: 12,
            physics: const NeverScrollableScrollPhysics(),
            children: [
              _quickActionTile('Anemia Check', Icons.camera_alt, () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CameraPage()),
                );
              }),
              _quickActionTile('History', Icons.history, () {}),
              _quickActionTile('Symptom Checker', Icons.health_and_safety, () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CameraPage()),
                );
              }),
              _quickActionTile('Saved Results', Icons.save, () {}),
            ],
          ),

          const SizedBox(height: 30),
          const Text('Upcoming Events', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          ...events.map((e) => _eventCard(e['title']!, e['date']!)),

          const SizedBox(height: 20),
          const Text('Latest News & Blogs', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          ...blogs.map((b) => _newsCard(b['title']!, b['summary']!)),
        ],
      ),
    );
  }
}