# AutoTrace Digitizer

**Current Version:** v1.0.0  
**Platform:** Windows portable release  
**Project Type:** Modified educational fork of Engauge Digitizer

## Overview

**AutoTrace Digitizer** is a modified version of Engauge Digitizer focused on reducing manual setup and repetitive point-clicking during graph digitizing.

This version keeps the core Engauge Digitizer workflow while adding automated tools for:

- Axis setup
- Curve-point placement
- Portable Windows use

The project is intended for classroom, research, instructional, and data-extraction contexts where users need to convert graph images into editable digitized data points more efficiently.

## Relationship to Engauge Digitizer

AutoTrace Digitizer is a modified fork of Engauge Digitizer.

This project is **not** the official upstream Engauge Digitizer release. The core interface and many core digitizing functions are based on the original Engauge Digitizer project. This version adds automation-focused features and a portable Windows packaging workflow.

Original license and attribution notices are preserved. New modifications are documented in this repository.

## What This Version Adds

### Auto Axis

The **Auto Axis** feature helps users begin graph calibration more quickly.

Current Auto Axis behavior includes:

- Detecting the bottom x-axis from an imported graph image
- Handling mild image tilt before calibration
- Creating normal editable axis points
- Setting the x-axis starting value to 0
- Setting the y-axis minimum to 0
- Prompting the user to enter the y-axis maximum

The generated axis points remain editable, so users can manually adjust calibration after automatic placement.

### Auto Curve

The **Auto Curve** feature helps place curve points automatically on the selected curve.

Current Auto Curve behavior includes:

- Detecting visible point markers inside the calibrated plot area
- Adding detected points to the selected curve
- Creating normal editable Engauge curve points
- Clamping detected y-values within the calibrated axis range

Auto Curve is intended to reduce repetitive manual clicking when graph images contain visible plotted markers.

## Interface Changes

AutoTrace Digitizer adds Auto Axis and Auto Curve access through the standard interface:

- Digitize menu
- Drawing toolbar

## Portable Windows Release

AutoTrace Digitizer v1.0.0 is distributed as a portable Windows package.

The portable build includes:

- AutoTrace/Engauge executable
- Required Qt runtime files
- Portable launcher support
- `portable-settings.ini`
- Local portable settings folder
- App-local runtime behavior

The application can be launched without installing the full development environment.

## Download

Download the current portable Windows release from the GitHub Releases page.

Recommended release asset:

```text
EngaugeDigitizer-Windows-Portable.zip
