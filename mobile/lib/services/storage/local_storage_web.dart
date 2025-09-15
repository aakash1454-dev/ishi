// lib/services/storage/local_storage_web.dart
class TestResult {
  int id = 0;
  late DateTime timestamp;
  late bool anemic;
  late double score;
  String? imagePath;
  String? notes;
}

class LocalStore {
  static final List<TestResult> _mem = [];

  static Future<int> addResult(TestResult r) async {
    r.id = _mem.length + 1;
    _mem.add(r);
    return r.id;
    }

  static Future<List<TestResult>> listResults({int limit = 100}) async {
    final list = List<TestResult>.from(_mem);
    list.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return list.take(limit).toList();
  }

  static Future<void> clearAll() async {
    _mem.clear();
  }
}
