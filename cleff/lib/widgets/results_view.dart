import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:just_audio/just_audio.dart';
import 'package:glass_kit/glass_kit.dart';
import 'package:path_provider/path_provider.dart';
import 'package:open_file/open_file.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../main.dart';
import 'stem_player_widget.dart';
import 'dart:html' as html;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:convert';

class ResultsView extends StatefulWidget {
  final List<Stem> results;
  final VoidCallback onReset;

  const ResultsView({super.key, required this.results, required this.onReset});

  @override
  State<ResultsView> createState() => _ResultsViewState();
}

class _ResultsViewState extends State<ResultsView> {
  late final List<AudioPlayer> _players;
  bool _isMasterPlaying = false;
  List<bool> _mutes = [];

  final sheetMusicEligible = const [
    'guitar',
    'piano',
    'bass',
    'vocals',
    'other',
  ];

  @override
  void initState() {
    super.initState();
    _players = widget.results.map((stem) {
      final url = 'http://api.aryanjumani.com/${stem.path}';
      return AudioPlayer()..setUrl(url);
    }).toList();

    _mutes = List.generate(widget.results.length, (_) => false);

    if (_players.isNotEmpty) {
      _players.first.playerStateStream.listen((state) {
        if (mounted) setState(() => _isMasterPlaying = state.playing);
      });
    }
  }

  @override
  void dispose() {
    for (final p in _players) p.dispose();
    super.dispose();
  }

  void _toggleMasterPlay() {
    if (_isMasterPlaying) {
      for (final p in _players) p.pause();
    } else {
      for (final p in _players) p.play();
    }
  }

  void _toggleMute(int i) {
    setState(() {
      _mutes[i] = !_mutes[i];
      _players[i].setVolume(_mutes[i] ? 0 : 1);
    });
  }

  void _seekAll(Duration d) {
    for (final p in _players) p.seek(d);
  }

  Future<void> downloadFileWeb(Uint8List bytes, String filename) async {
    final blob = html.Blob([bytes]);
    final url = html.Url.createObjectUrlFromBlob(blob);
    final anchor = html.AnchorElement(href: url)
      ..setAttribute("download", filename)
      ..click();
    html.Url.revokeObjectUrl(url);
  }

  Future<void> _downloadStem(Stem stem) async {
    try {
      final url = 'http://api.aryanjumani.com/${stem.path}';
      final response = await http.get(Uri.parse(url));
      if (kIsWeb) {
        await downloadFileWeb(response.bodyBytes, '${stem.name}.mp3');
      } else {
        final dir = await getApplicationDocumentsDirectory();
        final file = File('${dir.path}/${stem.name}.mp3');
        await file.writeAsBytes(response.bodyBytes);
      }
    } catch (e) {
      debugPrint('Download error: $e');
    }
  }

  Future<void> _generateSheetMusic(Stem stem) async {
    // Show a loading dialog for better user experience
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return const Dialog(
          child: Padding(
            padding: EdgeInsets.all(20.0),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircularProgressIndicator(),
                SizedBox(width: 20),
                Text("Generating file..."),
              ],
            ),
          ),
        );
      },
    );

    try {
      final url = 'http://api.aryanjumani.com/transcribe?stem=${stem.path}';
      final response = await http.get(Uri.parse(url));

      Navigator.pop(context); // Close the loading dialog

      if (response.statusCode != 200) {
        debugPrint('Sheet generation failed: ${response.statusCode}');
        return;
      }

      // This is the original download logic
      if (kIsWeb) {
        // The server returns a JSON object, so we need to decode it first
        final Map<String, dynamic> data = json.decode(response.body);
        final String xmlContent = data['xml'] ?? '';
        // Convert the string content to bytes for download
        final Uint8List fileBytes = utf8.encode(xmlContent);
        await downloadFileWeb(fileBytes, '${stem.name}.musicxml');
      } else {
        final Map<String, dynamic> data = json.decode(response.body);
        final String xmlContent = data['xml'] ?? '';
        final Uint8List fileBytes = utf8.encode(xmlContent);

        final dir = await getApplicationDocumentsDirectory();
        final file = File('${dir.path}/${stem.name}.musicxml');
        await file.writeAsBytes(fileBytes);
        OpenFile.open(file.path);
      }
    } catch (e) {
      Navigator.pop(context); // Close dialog on error
      debugPrint('Sheet generation error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_players.isEmpty) {
      return Column(
        children: [
          const SizedBox(height: 32),
          const Text("No results available", style: TextStyle(fontSize: 18)),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: widget.onReset,
            icon: const Icon(FontAwesomeIcons.rotateRight),
            label: const Text('Try Again'),
          ),
        ],
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          const SizedBox(height: 16),
          Text(
            "Separation Complete!",
            style: Theme.of(
              context,
            ).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 6),
          const Text(
            "Listen, download, or generate sheet music for each stem.",
            style: TextStyle(color: Colors.white70, fontSize: 15),
          ),
          const SizedBox(height: 28),

          // Master Controls
          GlassContainer.clearGlass(
            height: 90,
            width: double.infinity,
            borderRadius: BorderRadius.circular(20),
            gradient: LinearGradient(
              colors: [
                Colors.white.withOpacity(0.2),
                Colors.white.withOpacity(0.05),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderColor: Colors.white24,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  IconButton(
                    iconSize: 48,
                    color: Theme.of(context).primaryColor,
                    icon: Icon(
                      _isMasterPlaying
                          ? FontAwesomeIcons.circlePause
                          : FontAwesomeIcons.circlePlay,
                    ),
                    onPressed: _toggleMasterPlay,
                  ),
                  Expanded(
                    child: StreamBuilder<Duration>(
                      stream: _players.first.positionStream,
                      builder: (context, snapshot) {
                        final pos = snapshot.data ?? Duration.zero;
                        final dur = _players.first.duration ?? Duration.zero;
                        return Slider(
                          value: pos.inMilliseconds
                              .clamp(0, dur.inMilliseconds)
                              .toDouble(),
                          max: dur.inMilliseconds.toDouble(),
                          onChanged: (v) =>
                              _seekAll(Duration(milliseconds: v.toInt())),
                          activeColor: Theme.of(context).primaryColor,
                          inactiveColor: Colors.white30,
                        );
                      },
                    ),
                  ),
                  IconButton(
                    icon: const Icon(FontAwesomeIcons.rotateRight, size: 24),
                    color: Colors.white70,
                    tooltip: 'Redo (Replay from start)',
                    onPressed: () => _seekAll(Duration.zero),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 28),

          // Stems using StemPlayerWidget
          ...List.generate(widget.results.length, (i) {
            final stem = widget.results[i];
            return Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: StemPlayerWidget(
                stem: stem,
                player: _players[i],
                isMuted: _mutes[i],
                onMute: () => _toggleMute(i),
                onDownload: () => _downloadStem(stem),
                onGenerateSheet:
                    sheetMusicEligible.contains(stem.name.toLowerCase())
                    ? () => _generateSheetMusic(stem)
                    : null,
              ),
            );
          }),

          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: widget.onReset,
            icon: const Icon(FontAwesomeIcons.rotateRight),
            label: const Text("Separate Another File"),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 16),
              backgroundColor: Theme.of(context).primaryColor.withOpacity(0.25),
              foregroundColor: Theme.of(context).primaryColor,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
              side: BorderSide(color: Theme.of(context).primaryColor),
            ),
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }
}
