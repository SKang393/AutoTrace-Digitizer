# AutoTrace Digitizer v1.0.0 Release Notes

AutoTrace Digitizer is a modified fork of Engauge Digitizer. It is not the official upstream Engauge Digitizer release. Original Engauge Digitizer copyright and license notices are preserved.

## Added

- Windows portable release workflow.
- Auto Axis button in the Digitize menu and drawing toolbar.
- Auto Curve button in the Digitize menu and drawing toolbar.
- Portable settings support through `portable-settings.ini`, `ENGAUGE_PORTABLE=1`, and a local `settings` folder.
- US-English default locale unless the user selects another locale.
- Unicode-safe portable path handling for non-English Windows folders.

## Auto Axis

Auto Axis detects the bottom x-axis, supports mild image tilt correction, creates editable axis points, sets x start to 1, sets y minimum to 0, and prompts for y maximum.

## Auto Curve

Auto Curve now detects marker objects rather than raw foreground pixels. It rejects axes, gridlines, phase dividers, labels, and text boxes, places one editable curve point at each detected marker center, groups visually matching markers, and cycles detected marker groups on repeated clicks.

## Limitations

- Auto Axis does not read axis numbers automatically.
- Auto Curve is experimental. It is designed to reject common single-case design artifacts, but it may still need correction on low-resolution images, dense gridlines, overlapping labels, unusual marker symbols, or connecting lines touching markers.
- Manual review of digitized points is recommended before using exported data.
