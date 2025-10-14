import 'package:cleff/widgets/file_upload_view.dart';
import 'package:cleff/widgets/processing_view.dart';
import 'package:cleff/widgets/results_view.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

enum AppState { idle, processing, complete }

class Stem {
  final String name;
  final String path;
  Stem({required this.name, required this.path});
}

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cleff',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        fontFamily: 'Inter',
        scaffoldBackgroundColor: const Color(0xFF111119),
        primaryColor: const Color(0xFFC026D3), // Fuchsia
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFFC026D3), // Fuchsia
          secondary: Color(0xFF4ADE80), // Green
          surface: Color(0xFF1E1B2E), // Card background
          onSurface: Color(0xFFE2E0E4), // Main text
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  AppState _appState = AppState.idle;
  Stream<String> _logStream = const Stream.empty();
  List<Stem> _results = [];
  String _fileName = '';

  void _handleReset() {
    setState(() {
      _appState = AppState.idle;
      _logStream = const Stream.empty();
      _results = [];
      _fileName = '';
    });
  }

  Widget _buildCurrentView() {
    switch (_appState) {
      case AppState.processing:
        return ProcessingView(
          logStream: _logStream,
          fileName: _fileName,
          key: const ValueKey('processing'),
        );
      case AppState.complete:
        return ResultsView(
          results: _results,
          onReset: _handleReset,
          key: const ValueKey('results'),
        );
      case AppState.idle:
      default:
        return FileUploadView(
          key: const ValueKey('upload'),
          onUpload: (fileName, logStream, resultsFuture) {
            setState(() {
              _appState = AppState.processing;
              _fileName = fileName;
              _logStream = logStream;
            });
            resultsFuture
                .then((stems) {
                  setState(() {
                    _appState = AppState.complete;
                    _results = stems;
                  });
                })
                .catchError((e) {
                  // Handle potential errors during upload/processing
                  print("Error during processing: $e");
                  _handleReset();
                });
          },
        );
    }
  }

  // In main.dart -> class _HomePageState

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: RadialGradient(
            center: Alignment.center,
            radius: 0.8,
            colors: [Color(0xFF2E2257), Color(0xFF111119)],
          ),
        ),
        // We make the whole body a scroll view
        child: SingleChildScrollView(
          // We use a centered box that is at least as tall as the screen
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 700),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // Your Icon/SvgPicture widget will go here
                        FaIcon(
                          FontAwesomeIcons.codeFork,
                          color: Color(0xFFC026D3),
                          size: 36,
                        ),
                        const SizedBox(width: 12),
                        const SelectableText(
                          'Cleff',
                          style: TextStyle(
                            fontSize: 52,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const SelectableText(
                      'AI-powered audio separation for creators. Isolate vocals, drums, and instruments with studio-quality results.',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 18, color: Color(0xFFA1A1AA)),
                    ),
                    const SizedBox(height: 40),
                    SelectionArea(
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 400),
                        transitionBuilder: (child, animation) {
                          return FadeTransition(
                            opacity: animation,
                            child: child,
                          );
                        },
                        child: _buildCurrentView(),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
