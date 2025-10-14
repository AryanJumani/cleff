import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:dotted_border/dotted_border.dart';
import 'package:glass_kit/glass_kit.dart';
import 'package:http/http.dart' as http;
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../main.dart';

class FileUploadView extends StatefulWidget {
  final Function(String, Stream<String>, Future<List<Stem>>) onUpload;
  const FileUploadView({super.key, required this.onUpload});

  @override
  State<FileUploadView> createState() => _FileUploadViewState();
}

class _FileUploadViewState extends State<FileUploadView> {
  PlatformFile? _selectedFile;
  bool _isHovering = false;

  Future<void> _pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.audio,
    );
    if (result != null) setState(() => _selectedFile = result.files.first);
  }

  void _handleUpload() {
    if (_selectedFile == null) return;
    final streamController = StreamController<String>.broadcast();
    final completer = Completer<List<Stem>>();
    widget.onUpload(
      _selectedFile!.name,
      streamController.stream,
      completer.future,
    );
    _uploadFile(streamController, completer);
  }

  Future<void> _uploadFile(
    StreamController<String> log,
    Completer<List<Stem>> completer,
  ) async {
    const apiUrl = 'http://127.0.0.1:8000/separate-live';
    var request = http.MultipartRequest('POST', Uri.parse(apiUrl))
      ..files.add(
        http.MultipartFile.fromBytes(
          'file',
          _selectedFile!.bytes!,
          filename: _selectedFile!.name,
        ),
      );

    try {
      log.add("Uploading...");
      final response = await request.send();
      if (response.statusCode == 200) {
        log.add("Analyzing audio...");
        response.stream.transform(utf8.decoder).listen((value) {
          final lines = value
              .split('\n\n')
              .where((line) => line.startsWith('data:'));
          for (final line in lines) {
            final data = line.substring(5).trim();
            try {
              final jsonData = jsonDecode(data);
              if (jsonData['status'] == 'done') {
                log.add("Separation complete!");
                final stems = (jsonData['paths'] as Map<String, dynamic>)
                    .entries
                    .map((e) => Stem(name: e.key, path: e.value))
                    .toList();
                completer.complete(stems);
                log.close();
              }
            } catch (_) {
              log.add(data);
            }
          }
        });
      } else {
        throw Exception('Failed to upload file');
      }
    } catch (e) {
      log.add("Error: ${e.toString()}");
      log.close();
      completer.completeError(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    return GlassContainer(
      height: 450,
      width: double.infinity,
      gradient: LinearGradient(
        colors: [
          Colors.white.withOpacity(0.05),
          Colors.white.withOpacity(0.02),
        ],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      borderGradient: LinearGradient(
        colors: [Colors.white.withOpacity(0.2), Colors.white.withOpacity(0.1)],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      borderColor: Colors.white.withOpacity(0.1),
      blur: 15,
      borderRadius: BorderRadius.circular(24),
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            if (_selectedFile == null)
              MouseRegion(
                onEnter: (_) => setState(() => _isHovering = true),
                onExit: (_) => setState(() => _isHovering = false),
                child: GestureDetector(
                  onTap: _pickFile,
                  child: DottedBorder(
                    options: RoundedRectDottedBorderOptions(
                      color: _isHovering
                          ? Theme.of(context).primaryColor
                          : Colors.white.withOpacity(0.3),
                      strokeWidth: 2,
                      dashPattern: const [8, 6],
                      radius: const Radius.circular(16),
                    ),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      height: 200,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: _isHovering
                            ? Theme.of(context).primaryColor.withOpacity(0.1)
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            FontAwesomeIcons.cloudArrowUp,
                            size: 48,
                            color: Colors.white70,
                          ),
                          SizedBox(height: 16),
                          SelectableText(
                            'Drag & drop or click to upload',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.white70,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              )
            else
              Column(
                children: [
                  const Icon(
                    FontAwesomeIcons.fileAudio,
                    size: 80,
                    color: Color(0xFFC026D3),
                  ),
                  const SizedBox(height: 24),
                  SelectableText(
                    _selectedFile!.name,
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  SelectableText(
                    '${(_selectedFile!.size / 1024 / 1024).toStringAsFixed(2)} MB',
                    style: const TextStyle(fontSize: 16, color: Colors.white70),
                  ),
                  const SizedBox(height: 32),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      ElevatedButton.icon(
                        onPressed: _handleUpload,
                        icon: const Icon(FontAwesomeIcons.wandMagicSparkles),
                        label: const Text('Separate Stems'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 32,
                            vertical: 16,
                          ),
                          backgroundColor: Theme.of(context).primaryColor,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30),
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      IconButton(
                        onPressed: () => setState(() => _selectedFile = null),
                        icon: const Icon(FontAwesomeIcons.x),
                        tooltip: 'Remove file',
                      ),
                    ],
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
