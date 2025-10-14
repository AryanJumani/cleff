import 'dart:async';
import 'package:flutter/material.dart';
import 'package:glass_kit/glass_kit.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

enum StepStatus { waiting, active, complete, error }

class ProcessStep {
  final String title;
  final IconData icon;
  final List<String> triggers;
  StepStatus status;

  ProcessStep({
    required this.title,
    required this.icon,
    required this.triggers,
    this.status = StepStatus.waiting,
  });
}

class ProcessingView extends StatefulWidget {
  final Stream<String> logStream;
  final String fileName;

  const ProcessingView({
    super.key,
    required this.logStream,
    required this.fileName,
  });

  @override
  State<ProcessingView> createState() => _ProcessingViewState();
}

class _ProcessingViewState extends State<ProcessingView>
    with TickerProviderStateMixin {
  late final List<ProcessStep> _steps;
  late StreamSubscription<String> _logSubscription;
  int _activeIndex = 0;

  @override
  void initState() {
    super.initState();
    _steps = [
      ProcessStep(
        title: "Uploading",
        icon: FontAwesomeIcons.cloudArrowUp,
        triggers: ["Uploading..."],
      ),
      ProcessStep(
        title: "Analyzing Audio",
        icon: FontAwesomeIcons.chartBar,
        triggers: ["Analyzing audio...", "Loading model..."],
      ),
      ProcessStep(
        title: "Separating Stems",
        icon: FontAwesomeIcons.codeFork,
        triggers: ["Separating..."],
      ),
      ProcessStep(
        title: "Complete",
        icon: FontAwesomeIcons.circleCheck,
        triggers: ["Separation complete!"],
      ),
    ];

    _logSubscription = widget.logStream.listen((log) {
      if (!mounted) return;
      setState(() {
        for (int i = 0; i < _steps.length; i++) {
          if (_steps[i].triggers.any(
            (trigger) => log.toLowerCase().contains(trigger.toLowerCase()),
          )) {
            // Mark previous steps as complete
            for (int j = 0; j < i; j++) {
              _steps[j].status = StepStatus.complete;
            }
            _steps[i].status = StepStatus.active;
            _activeIndex = i;
            if (log.toLowerCase().contains("error")) {
              _steps[i].status = StepStatus.error;
            }
            break;
          }
        }
      });
    });
  }

  @override
  void dispose() {
    _logSubscription.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GlassContainer(
      height: 480,
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
          children: [
            SelectableText(
              "Crafting your stems...",
              style: Theme.of(
                context,
              ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            SelectableText(
              widget.fileName,
              style: TextStyle(color: Colors.white.withOpacity(0.7)),
              maxLines: 1,
            ),
            const SizedBox(height: 40),
            ...List.generate(_steps.length, (index) {
              return ProcessStepTile(
                step: _steps[index],
                isActive: index == _activeIndex,
                isLast: index == _steps.length - 1,
              );
            }),
          ],
        ),
      ),
    );
  }
}

class ProcessStepTile extends StatefulWidget {
  final ProcessStep step;
  final bool isActive;
  final bool isLast;

  const ProcessStepTile({
    super.key,
    required this.step,
    required this.isActive,
    required this.isLast,
  });

  @override
  State<ProcessStepTile> createState() => _ProcessStepTileState();
}

class _ProcessStepTileState extends State<ProcessStepTile>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    _animation = CurvedAnimation(parent: _controller, curve: Curves.easeInOut);
    if (widget.step.status == StepStatus.active ||
        widget.step.status == StepStatus.complete) {
      _controller.forward();
    }
  }

  @override
  void didUpdateWidget(covariant ProcessStepTile oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.step.status == StepStatus.active ||
        widget.step.status == StepStatus.complete) {
      _controller.forward();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Color activeColor = Theme.of(context).primaryColor;
    Color inactiveColor = Colors.white.withOpacity(0.4);
    Color iconColor;
    Widget icon;

    switch (widget.step.status) {
      case StepStatus.active:
        iconColor = activeColor;
        icon = SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            valueColor: AlwaysStoppedAnimation<Color>(activeColor),
          ),
        );
        break;
      case StepStatus.complete:
        iconColor = Theme.of(context).colorScheme.secondary;
        icon = Icon(FontAwesomeIcons.circleCheck, color: iconColor);
        break;
      case StepStatus.error:
        iconColor = Colors.redAccent;
        icon = Icon(FontAwesomeIcons.circleXmark, color: iconColor);
        break;
      case StepStatus.waiting:
      default:
        iconColor = inactiveColor;
        icon = Icon(widget.step.icon, color: iconColor);
        break;
    }

    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Column(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: iconColor, width: 2),
                ),
                child: icon,
              ),
              if (!widget.isLast)
                Expanded(
                  child: Container(
                    width: 2,
                    color: inactiveColor,
                    child: Align(
                      alignment: Alignment.topCenter,
                      child: SizeTransition(
                        sizeFactor: _animation,
                        child: Container(
                          width: 2,
                          color: Theme.of(context).colorScheme.secondary,
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(width: 24),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(top: 12.0, bottom: 32),
              child: SelectableText(
                widget.step.title,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: widget.step.status == StepStatus.waiting
                      ? inactiveColor
                      : Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
