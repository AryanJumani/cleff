import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:glass_kit/glass_kit.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../main.dart';

class StemPlayerWidget extends StatelessWidget {
  final Stem stem;
  final AudioPlayer player;
  final bool isMuted;
  final VoidCallback onMute;
  final Future<void> Function() onDownload;
  final Future<void> Function()? onGenerateSheet;

  const StemPlayerWidget({
    super.key,
    required this.stem,
    required this.player,
    required this.isMuted,
    required this.onMute,
    required this.onDownload,
    required this.onGenerateSheet,
  });

  @override
  Widget build(BuildContext context) {
    final nameCapital = stem.name[0].toUpperCase() + stem.name.substring(1);

    return GlassContainer.clearGlass(
      height: 120,
      width: double.infinity,
      borderRadius: BorderRadius.circular(24),
      gradient: LinearGradient(
        colors: [
          Colors.white.withOpacity(0.15),
          Colors.white.withOpacity(0.05),
        ],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      borderColor: Colors.white24,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            GestureDetector(
              onTap: onMute,
              child: Row(
                children: [
                  Icon(
                    isMuted
                        ? FontAwesomeIcons.volumeXmark
                        : FontAwesomeIcons.volumeHigh,
                    color: isMuted
                        ? Colors.white38
                        : Theme.of(context).primaryColor,
                  ),
                  const SizedBox(width: 10),
                  Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        nameCapital,
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: isMuted ? Colors.white38 : Colors.white,
                        ),
                      ),
                      Text(
                        isMuted ? "Muted" : "Active",
                        style: TextStyle(
                          fontSize: 14,
                          color: isMuted ? Colors.white38 : Colors.white70,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            Row(
              children: [
                IconButton(
                  icon: const Icon(FontAwesomeIcons.download),
                  color: Colors.white70,
                  tooltip: 'Download Stem',
                  onPressed: () async => await onDownload(),
                ),
                if (onGenerateSheet != null)
                  IconButton(
                    icon: const Icon(FontAwesomeIcons.music),
                    color: Colors.white70,
                    tooltip: 'Generate Sheet Music',
                    onPressed: () async => await onGenerateSheet!(),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
