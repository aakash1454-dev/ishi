// lib/services/storage/local_storage.dart
import 'dart:async';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:isar/isar.dart';
import 'package:path_provider/path_provider.dart';

part 'local_storage.g.dart';

@collection
class TestResult {
  Id id = Isar.autoIncrement;
  late DateTime timestamp;
  late bool anemic;
  late double score;
  String? imagePath; // optional path to saved input/crop
  String? notes;     // optional notes
}

class LocalStore {
  static Isar? _isar;

    static Future<Isar> instance() async {
    if (_isar != null) return _isar!;

    if (kIsWeb) {
      // Some analyzer/toolchains require a directory arg—use a dummy value on web.
      _isar = await Isar.open(
        [TestResultSchema],
        directory: 'isar_web',
      );
    } else {
      final dir = await getApplicationDocumentsDirectory();
      _isar = await Isar.open(
        [TestResultSchema],
        directory: dir.path,
      );
    }
    return _isar!;
  }


  static Future<int> addResult(TestResult r) async {
    final isar = await instance();
    return isar.writeTxn(() => isar.testResults.put(r));
  }

  static Future<List<TestResult>> listResults({int limit = 100}) async {
    final isar = await instance();
    return isar.testResults
        .where()
        .sortByTimestampDesc()
        .limit(limit)
        .findAll();
  }

  static Future<void> clearAll() async {
    final isar = await instance();
    await isar.writeTxn(() async {
      await isar.testResults.clear();
    });
  }
}
