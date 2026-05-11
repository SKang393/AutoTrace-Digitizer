/******************************************************************************************************
 * (C) 2026. This file is part of Engauge Digitizer, which is released under GNU General Public       *
 * License version 2 (GPLv2) or (at your option) any later version. See file LICENSE for details.     *
 ******************************************************************************************************/

#include "AutoDigitize.h"
#include "Transformation.h"
#include <QColor>
#include <QImage>
#include <QObject>
#include <QPainter>
#include <QQueue>
#include <QTransform>
#include <QVector>
#include <QtGlobal>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

const double AXIS_ROTATION_THRESHOLD_DEGREES = 0.7;
const double MAX_AXIS_ROTATION_DEGREES = 8.0;
const double PI = 3.14159265358979323846;
const int CURVE_POINT_MIN_SPACING = 6;
const int MARKER_PATCH_SIZE = 32;
const int MARKER_PATCH_FEATURE_SIZE = 16;
const double MARKER_CLUSTER_DISTANCE_THRESHOLD = 0.19;

struct Run
{
  Run() :
    start (0),
    end (0),
    rowOrColumn (0),
    length (0)
  {
  }

  int start;
  int end;
  int rowOrColumn;
  int length;
};

struct Component
{
  Component() :
    area (0),
    sumX (0),
    sumY (0),
    minX (std::numeric_limits<int>::max()),
    maxX (std::numeric_limits<int>::min()),
    minY (std::numeric_limits<int>::max()),
    maxY (std::numeric_limits<int>::min())
  {
  }

  int area;
  long long sumX;
  long long sumY;
  int minX;
  int maxX;
  int minY;
  int maxY;
};

struct MarkerCandidate
{
  QPoint center;
  QRect bounds;
  int area;
  double size;
  QVector<unsigned char> patch;
  QVector<double> features;
};

struct MarkerCluster
{
  QList<int> candidateIndexes;
  QVector<double> centroid;
  double meanDistance;
  double sizeVariation;
};

QPoint clampPointToYRange(const QPointF &pointScreen,
                          const Transformation &transformation,
                          double yMinimum,
                          double yMaximum);

void addCurvePointIfSeparated(QList<QPoint> &points,
                              const QPoint &point);

inline int indexFor(int x,
                    int y,
                    int width)
{
  return y * width + x;
}

bool pixelIsForeground(QRgb rgb)
{
  if (qAlpha(rgb) < 20) {
    return false;
  }

  const int red = qRed(rgb);
  const int green = qGreen(rgb);
  const int blue = qBlue(rgb);
  const int maximum = qMax(red, qMax(green, blue));
  const int minimum = qMin(red, qMin(green, blue));
  const int gray = qGray(rgb);

  return (gray < 190) || ((maximum - minimum) > 45 && maximum < 248);
}

QVector<unsigned char> foregroundMask(const QImage &image)
{
  const QImage converted = image.convertToFormat(QImage::Format_ARGB32);
  const int width = converted.width();
  const int height = converted.height();
  QVector<unsigned char> mask(width * height, 0);

  for (int y = 0; y < height; ++y) {
    const QRgb *line = reinterpret_cast<const QRgb *> (converted.constScanLine(y));
    for (int x = 0; x < width; ++x) {
      if (pixelIsForeground(line [x])) {
        mask [indexFor(x, y, width)] = 1;
      }
    }
  }

  return mask;
}

Run longestHorizontalRun(const QVector<unsigned char> &mask,
                         int width,
                         int row,
                         int gapAllowed)
{
  Run best;
  best.rowOrColumn = row;

  int start = -1;
  int lastForeground = -1;
  int gap = 0;

  for (int x = 0; x < width; ++x) {
    if (mask [indexFor(x, row, width)]) {
      if (start < 0) {
        start = x;
      }
      lastForeground = x;
      gap = 0;
    } else if (start >= 0) {
      ++gap;
      if (gap > gapAllowed) {
        const int length = lastForeground - start + 1;
        if (length > best.length) {
          best.start = start;
          best.end = lastForeground;
          best.length = length;
        }
        start = -1;
        lastForeground = -1;
        gap = 0;
      }
    }
  }

  if (start >= 0) {
    const int length = lastForeground - start + 1;
    if (length > best.length) {
      best.start = start;
      best.end = lastForeground;
      best.length = length;
    }
  }

  return best;
}

Run longestVerticalRun(const QVector<unsigned char> &mask,
                       int width,
                       int height,
                       int column,
                       int yMaximum,
                       int gapAllowed)
{
  Run best;
  best.rowOrColumn = column;

  int start = -1;
  int lastForeground = -1;
  int gap = 0;
  const int yLimit = qBound(0, yMaximum, height - 1);

  for (int y = 0; y <= yLimit; ++y) {
    if (mask [indexFor(column, y, width)]) {
      if (start < 0) {
        start = y;
      }
      lastForeground = y;
      gap = 0;
    } else if (start >= 0) {
      ++gap;
      if (gap > gapAllowed) {
        const int length = lastForeground - start + 1;
        if (length > best.length) {
          best.start = start;
          best.end = lastForeground;
          best.length = length;
        }
        start = -1;
        lastForeground = -1;
        gap = 0;
      }
    }
  }

  if (start >= 0) {
    const int length = lastForeground - start + 1;
    if (length > best.length) {
      best.start = start;
      best.end = lastForeground;
      best.length = length;
    }
  }

  return best;
}

double estimateHorizontalAngleDegrees(const QImage &image)
{
  const QImage converted = image.convertToFormat(QImage::Format_ARGB32);
  const int width = converted.width();
  const int height = converted.height();
  const int step = qMax(1, qMax(width, height) / 800);
  QList<QPoint> points;

  for (int y = 0; y < height; y += step) {
    const QRgb *line = reinterpret_cast<const QRgb *> (converted.constScanLine(y));
    for (int x = 0; x < width; x += step) {
      if (pixelIsForeground(line [x])) {
        points << QPoint(x, y);
      }
    }
  }

  if (points.count() < 50) {
    return 0.0;
  }

  double bestAngle = 0.0;
  double bestScore = -1.0;

  for (int angleHalfDegrees = -16; angleHalfDegrees <= 16; ++angleHalfDegrees) {
    const double angleDegrees = angleHalfDegrees * 0.5;
    const double slope = std::tan(angleDegrees * PI / 180.0);
    const int binMin = static_cast<int> (std::floor(-std::abs(slope) * width)) - 4;
    const int binMax = height + static_cast<int> (std::ceil(std::abs(slope) * width)) + 4;
    const int binCount = binMax - binMin + 1;
    QVector<int> bins(binCount, 0);

    for (int index = 0; index < points.count(); ++index) {
      const QPoint point = points.at(index);
      const int bin = qRound((point.y() - slope * point.x()) / 2.0) - binMin;
      if (bin >= 0 && bin < binCount) {
        ++bins [bin];
      }
    }

    for (int bin = 0; bin < binCount; ++bin) {
      const double yIntercept = (bin + binMin) * 2.0;
      const double yCenter = yIntercept + slope * width / 2.0;
      const double bottomPreference = qBound(0.0, yCenter / qMax(1, height), 1.0);
      const double score = bins [bin] + bottomPreference * 3.0;
      if (score > bestScore) {
        bestScore = score;
        bestAngle = angleDegrees;
      }
    }
  }

  return bestAngle;
}

QImage rotateImage(const QImage &image,
                   double degrees)
{
  QTransform transform;
  transform.rotate(degrees);
  QRect rotatedRect = transform.mapRect(QRect(QPoint(0, 0), image.size()));
  QImage rotated(rotatedRect.size(), QImage::Format_ARGB32);
  rotated.fill(Qt::white);

  QPainter painter(&rotated);
  painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
  painter.translate(-rotatedRect.topLeft());
  painter.setTransform(transform, true);
  painter.drawImage(0, 0, image.convertToFormat(QImage::Format_ARGB32));
  painter.end();

  return rotated;
}

bool detectHorizontalAxis(const QVector<unsigned char> &mask,
                          int width,
                          int height,
                          Run &axisRun)
{
  Run best;
  const int minimumRunLength = qMax(40, width / 5);

  for (int y = 0; y < height; ++y) {
    const Run run = longestHorizontalRun(mask, width, y, 4);
    if (run.length < minimumRunLength) {
      continue;
    }

    if (run.length > best.length ||
        (run.length >= best.length * 8 / 10 && y > best.rowOrColumn)) {
      best = run;
    }
  }

  if (best.length < minimumRunLength) {
    return false;
  }

  axisRun = best;
  return true;
}

bool detectVerticalAxis(const QVector<unsigned char> &mask,
                        int width,
                        int height,
                        const Run &xAxisRun,
                        Run &axisRun)
{
  Run best;
  double bestScore = -1.0;
  const int yMaximum = xAxisRun.rowOrColumn;
  const int minimumRunLength = qMax(30, height / 5);
  const int columnMaximum = qMin(width - 1, qMax(xAxisRun.start + width / 8, width / 2));

  for (int x = 0; x <= columnMaximum; ++x) {
    const Run run = longestVerticalRun(mask, width, height, x, yMaximum, 4);
    if (run.length < minimumRunLength) {
      continue;
    }

    const int bottomDistance = std::abs(run.end - yMaximum);
    const int leftDistance = std::abs(x - xAxisRun.start);
    const double score = run.length - bottomDistance * 0.7 - leftDistance * 0.08;
    if (score > bestScore) {
      best = run;
      bestScore = score;
    }
  }

  if (bestScore < 0.0) {
    return false;
  }

  axisRun = best;
  return true;
}

void addComponentPixel(Component &component,
                       int x,
                       int y)
{
  ++component.area;
  component.sumX += x;
  component.sumY += y;
  component.minX = qMin(component.minX, x);
  component.maxX = qMax(component.maxX, x);
  component.minY = qMin(component.minY, y);
  component.maxY = qMax(component.maxY, y);
}

QVector<unsigned char> denseForegroundMask(const QVector<unsigned char> &mask,
                                           int width,
                                           int height,
                                           const QRect &rect)
{
  QVector<unsigned char> dense(width * height, 0);
  const QRect bounds = rect.intersected(QRect(0, 0, width, height));

  for (int y = bounds.top(); y <= bounds.bottom(); ++y) {
    for (int x = bounds.left(); x <= bounds.right(); ++x) {
      if (!mask [indexFor(x, y, width)]) {
        continue;
      }

      int count = 0;
      for (int yy = qMax(bounds.top(), y - 2); yy <= qMin(bounds.bottom(), y + 2); ++yy) {
        for (int xx = qMax(bounds.left(), x - 2); xx <= qMin(bounds.right(), x + 2); ++xx) {
          count += mask [indexFor(xx, yy, width)];
        }
      }

      if (count >= 3) {
        dense [indexFor(x, y, width)] = 1;
      }
    }
  }

  return dense;
}

QVector<unsigned char> suppressLongRuns(const QVector<unsigned char> &mask,
                                        int width,
                                        int height,
                                        const QRect &rect)
{
  QVector<unsigned char> filtered = mask;
  const QRect bounds = rect.intersected(QRect(0, 0, width, height));
  const int horizontalMinimum = qMax(24, bounds.width() / 5);
  const int verticalMinimum = qMax(24, bounds.height() / 5);

  for (int y = bounds.top(); y <= bounds.bottom(); ++y) {
    int runStart = -1;
    for (int x = bounds.left(); x <= bounds.right() + 1; ++x) {
      const bool foreground = (x <= bounds.right()) && mask [indexFor(x, y, width)];
      if (foreground && runStart < 0) {
        runStart = x;
      } else if ((!foreground || x > bounds.right()) && runStart >= 0) {
        const int runEnd = x - 1;
        if (runEnd - runStart + 1 >= horizontalMinimum) {
          for (int xx = runStart; xx <= runEnd; ++xx) {
            filtered [indexFor(xx, y, width)] = 0;
          }
        }
        runStart = -1;
      }
    }
  }

  for (int x = bounds.left(); x <= bounds.right(); ++x) {
    int runStart = -1;
    for (int y = bounds.top(); y <= bounds.bottom() + 1; ++y) {
      const bool foreground = (y <= bounds.bottom()) && mask [indexFor(x, y, width)];
      if (foreground && runStart < 0) {
        runStart = y;
      } else if ((!foreground || y > bounds.bottom()) && runStart >= 0) {
        const int runEnd = y - 1;
        if (runEnd - runStart + 1 >= verticalMinimum) {
          for (int yy = runStart; yy <= runEnd; ++yy) {
            filtered [indexFor(x, yy, width)] = 0;
          }
        }
        runStart = -1;
      }
    }
  }

  return filtered;
}

QVector<unsigned char> extractNormalizedPatch(const QVector<unsigned char> &mask,
                                              int width,
                                              int height,
                                              const Component &component)
{
  QVector<unsigned char> patch(MARKER_PATCH_SIZE * MARKER_PATCH_SIZE, 0);
  const double centerX = static_cast<double> (component.sumX) / component.area;
  const double centerY = static_cast<double> (component.sumY) / component.area;
  const int componentWidth = component.maxX - component.minX + 1;
  const int componentHeight = component.maxY - component.minY + 1;
  const double side = qMax(8.0,
                           static_cast<double> (qMax(componentWidth, componentHeight) + 6));

  for (int py = 0; py < MARKER_PATCH_SIZE; ++py) {
    const double sourceY = centerY + ((static_cast<double> (py) + 0.5) / MARKER_PATCH_SIZE - 0.5) * side;
    const int y = qRound(sourceY);
    if (y < 0 || y >= height) {
      continue;
    }

    for (int px = 0; px < MARKER_PATCH_SIZE; ++px) {
      const double sourceX = centerX + ((static_cast<double> (px) + 0.5) / MARKER_PATCH_SIZE - 0.5) * side;
      const int x = qRound(sourceX);
      if (x < 0 || x >= width) {
        continue;
      }

      if (mask [indexFor(x, y, width)]) {
        patch [indexFor(px, py, MARKER_PATCH_SIZE)] = 1;
      }
    }
  }

  return patch;
}

double patchForegroundRatio(const QVector<unsigned char> &patch)
{
  int foreground = 0;
  for (int index = 0; index < patch.count(); ++index) {
    foreground += patch.at(index);
  }

  return static_cast<double> (foreground) / qMax(1, patch.count());
}

double patchHoleRatio(const QVector<unsigned char> &patch)
{
  QVector<unsigned char> visited(patch.count(), 0);
  QQueue<QPoint> queue;

  for (int x = 0; x < MARKER_PATCH_SIZE; ++x) {
    if (!patch [indexFor(x, 0, MARKER_PATCH_SIZE)]) {
      queue.enqueue(QPoint(x, 0));
      visited [indexFor(x, 0, MARKER_PATCH_SIZE)] = 1;
    }
    if (!patch [indexFor(x, MARKER_PATCH_SIZE - 1, MARKER_PATCH_SIZE)]) {
      queue.enqueue(QPoint(x, MARKER_PATCH_SIZE - 1));
      visited [indexFor(x, MARKER_PATCH_SIZE - 1, MARKER_PATCH_SIZE)] = 1;
    }
  }

  for (int y = 0; y < MARKER_PATCH_SIZE; ++y) {
    if (!patch [indexFor(0, y, MARKER_PATCH_SIZE)] &&
        !visited [indexFor(0, y, MARKER_PATCH_SIZE)]) {
      queue.enqueue(QPoint(0, y));
      visited [indexFor(0, y, MARKER_PATCH_SIZE)] = 1;
    }
    if (!patch [indexFor(MARKER_PATCH_SIZE - 1, y, MARKER_PATCH_SIZE)] &&
        !visited [indexFor(MARKER_PATCH_SIZE - 1, y, MARKER_PATCH_SIZE)]) {
      queue.enqueue(QPoint(MARKER_PATCH_SIZE - 1, y));
      visited [indexFor(MARKER_PATCH_SIZE - 1, y, MARKER_PATCH_SIZE)] = 1;
    }
  }

  while (!queue.isEmpty()) {
    const QPoint point = queue.dequeue();
    const QPoint neighbors [4] = {
      QPoint(point.x() - 1, point.y()),
      QPoint(point.x() + 1, point.y()),
      QPoint(point.x(), point.y() - 1),
      QPoint(point.x(), point.y() + 1)
    };

    for (int index = 0; index < 4; ++index) {
      const QPoint neighbor = neighbors [index];
      if (neighbor.x() < 0 || neighbor.x() >= MARKER_PATCH_SIZE ||
          neighbor.y() < 0 || neighbor.y() >= MARKER_PATCH_SIZE) {
        continue;
      }

      const int neighborIndex = indexFor(neighbor.x(), neighbor.y(), MARKER_PATCH_SIZE);
      if (patch [neighborIndex] || visited [neighborIndex]) {
        continue;
      }

      visited [neighborIndex] = 1;
      queue.enqueue(neighbor);
    }
  }

  int interiorBackground = 0;
  for (int index = 0; index < patch.count(); ++index) {
    if (!patch.at(index) && !visited.at(index)) {
      ++interiorBackground;
    }
  }

  return static_cast<double> (interiorBackground) / patch.count();
}

double patchEdgeDensity(const QVector<unsigned char> &patch)
{
  int transitions = 0;
  int comparisons = 0;
  for (int y = 0; y < MARKER_PATCH_SIZE; ++y) {
    for (int x = 0; x < MARKER_PATCH_SIZE; ++x) {
      const unsigned char value = patch [indexFor(x, y, MARKER_PATCH_SIZE)];
      if (x + 1 < MARKER_PATCH_SIZE) {
        transitions += value != patch [indexFor(x + 1, y, MARKER_PATCH_SIZE)];
        ++comparisons;
      }
      if (y + 1 < MARKER_PATCH_SIZE) {
        transitions += value != patch [indexFor(x, y + 1, MARKER_PATCH_SIZE)];
        ++comparisons;
      }
    }
  }

  return static_cast<double> (transitions) / qMax(1, comparisons);
}

QVector<double> markerFeatures(const MarkerCandidate &candidate)
{
  QVector<double> features;
  const double foregroundRatio = patchForegroundRatio(candidate.patch);
  const double holeRatio = patchHoleRatio(candidate.patch);
  const double edgeDensity = patchEdgeDensity(candidate.patch);
  const double density = static_cast<double> (candidate.area) /
                         qMax(1, candidate.bounds.width() * candidate.bounds.height());
  const double aspect = static_cast<double> (qMin(candidate.bounds.width(), candidate.bounds.height())) /
                        qMax(1, qMax(candidate.bounds.width(), candidate.bounds.height()));

  double sumX = 0.0;
  double sumY = 0.0;
  double count = 0.0;
  for (int y = 0; y < MARKER_PATCH_SIZE; ++y) {
    for (int x = 0; x < MARKER_PATCH_SIZE; ++x) {
      if (candidate.patch [indexFor(x, y, MARKER_PATCH_SIZE)]) {
        sumX += x;
        sumY += y;
        count += 1.0;
      }
    }
  }

  const double centerX = count > 0.0 ? sumX / count : MARKER_PATCH_SIZE / 2.0;
  const double centerY = count > 0.0 ? sumY / count : MARKER_PATCH_SIZE / 2.0;
  double varX = 0.0;
  double varY = 0.0;
  double covariance = 0.0;
  QVector<double> radial(4, 0.0);
  QVector<double> radialCounts(4, 0.0);
  for (int y = 0; y < MARKER_PATCH_SIZE; ++y) {
    for (int x = 0; x < MARKER_PATCH_SIZE; ++x) {
      const double dx = x - centerX;
      const double dy = y - centerY;
      const double radius = std::sqrt(dx * dx + dy * dy);
      const int radialIndex = qBound(0,
                                     static_cast<int> (radius / (MARKER_PATCH_SIZE / 8.0)),
                                     radial.count() - 1);
      radialCounts [radialIndex] += 1.0;
      if (candidate.patch [indexFor(x, y, MARKER_PATCH_SIZE)]) {
        varX += dx * dx;
        varY += dy * dy;
        covariance += dx * dy;
        radial [radialIndex] += 1.0;
      }
    }
  }

  const double varianceScale = MARKER_PATCH_SIZE * MARKER_PATCH_SIZE;
  features << foregroundRatio * 1.4
           << holeRatio * 2.2
           << edgeDensity
           << density
           << aspect
           << (candidate.bounds.width() / qMax(1.0, candidate.size)) * 0.35
           << (candidate.bounds.height() / qMax(1.0, candidate.size)) * 0.35
           << ((centerX / MARKER_PATCH_SIZE) - 0.5)
           << ((centerY / MARKER_PATCH_SIZE) - 0.5)
           << (count > 0.0 ? varX / count / varianceScale : 0.0)
           << (count > 0.0 ? varY / count / varianceScale : 0.0)
           << (count > 0.0 ? covariance / count / varianceScale : 0.0);

  for (int index = 0; index < radial.count(); ++index) {
    features << (radialCounts.at(index) > 0.0 ? radial.at(index) / radialCounts.at(index) : 0.0);
  }

  const int cellSize = MARKER_PATCH_SIZE / MARKER_PATCH_FEATURE_SIZE;
  for (int cellY = 0; cellY < MARKER_PATCH_FEATURE_SIZE; ++cellY) {
    for (int cellX = 0; cellX < MARKER_PATCH_FEATURE_SIZE; ++cellX) {
      int foreground = 0;
      for (int yy = cellY * cellSize; yy < (cellY + 1) * cellSize; ++yy) {
        for (int xx = cellX * cellSize; xx < (cellX + 1) * cellSize; ++xx) {
          foreground += candidate.patch [indexFor(xx, yy, MARKER_PATCH_SIZE)];
        }
      }
      features << 0.45 * foreground / qMax(1, cellSize * cellSize);
    }
  }

  return features;
}

double featureDistance(const QVector<double> &left,
                       const QVector<double> &right)
{
  const int count = qMin(left.count(), right.count());
  if (count == 0) {
    return std::numeric_limits<double>::max();
  }

  double sum = 0.0;
  for (int index = 0; index < count; ++index) {
    const double delta = left.at(index) - right.at(index);
    sum += delta * delta;
  }

  return std::sqrt(sum / count);
}

QVector<MarkerCandidate> markerCandidatesFromComponents(const QVector<unsigned char> &mask,
                                                        int width,
                                                        int height,
                                                        const QRect &bounds,
                                                        const Transformation &transformation,
                                                        double yMinimum,
                                                        double yMaximum)
{
  QVector<MarkerCandidate> candidates;
  QVector<unsigned char> visited(width * height, 0);
  const int maxMarkerWidth = qMax(18, qMin(96, bounds.width() / 5));
  const int maxMarkerHeight = qMax(18, qMin(96, bounds.height() / 5));

  for (int y = bounds.top(); y <= bounds.bottom(); ++y) {
    for (int x = bounds.left(); x <= bounds.right(); ++x) {
      const int startIndex = indexFor(x, y, width);
      if (!mask [startIndex] || visited [startIndex]) {
        continue;
      }

      Component component;
      QQueue<QPoint> queue;
      queue.enqueue(QPoint(x, y));
      visited [startIndex] = 1;

      while (!queue.isEmpty()) {
        const QPoint point = queue.dequeue();
        addComponentPixel(component, point.x(), point.y());

        for (int yy = point.y() - 1; yy <= point.y() + 1; ++yy) {
          for (int xx = point.x() - 1; xx <= point.x() + 1; ++xx) {
            if (xx < bounds.left() || xx > bounds.right() ||
                yy < bounds.top() || yy > bounds.bottom()) {
              continue;
            }

            const int neighborIndex = indexFor(xx, yy, width);
            if (!mask [neighborIndex] || visited [neighborIndex]) {
              continue;
            }

            visited [neighborIndex] = 1;
            queue.enqueue(QPoint(xx, yy));
          }
        }
      }

      const int componentWidth = component.maxX - component.minX + 1;
      const int componentHeight = component.maxY - component.minY + 1;
      const double aspect = componentHeight == 0 ?
                            999.0 :
                            static_cast<double> (qMax(componentWidth, componentHeight)) /
                            qMax(1, qMin(componentWidth, componentHeight));
      const double density = static_cast<double> (component.area) /
                             qMax(1, componentWidth * componentHeight);

      if (component.area < 5 ||
          component.area > 2600 ||
          componentWidth < 3 ||
          componentHeight < 3 ||
          componentWidth > maxMarkerWidth ||
          componentHeight > maxMarkerHeight ||
          aspect > 4.2 ||
          density < 0.08) {
        continue;
      }

      const QPointF centerRaw(static_cast<double> (component.sumX) / component.area,
                              static_cast<double> (component.sumY) / component.area);
      const QPoint center = clampPointToYRange(centerRaw,
                                               transformation,
                                               yMinimum,
                                               yMaximum);
      if (!bounds.contains(center)) {
        continue;
      }

      MarkerCandidate candidate;
      candidate.center = center;
      candidate.bounds = QRect(QPoint(component.minX, component.minY),
                               QPoint(component.maxX, component.maxY));
      candidate.area = component.area;
      candidate.size = qMax(componentWidth, componentHeight);
      candidate.patch = extractNormalizedPatch(mask,
                                               width,
                                               height,
                                               component);
      candidate.features = markerFeatures(candidate);
      candidates << candidate;
    }
  }

  return candidates;
}

bool pointIsNearExistingCandidate(const QPoint &point,
                                  const QVector<MarkerCandidate> &candidates,
                                  int minimumDistance)
{
  const int minimumDistanceSquared = minimumDistance * minimumDistance;
  for (int index = 0; index < candidates.count(); ++index) {
    const QPoint delta = candidates.at(index).center - point;
    if (delta.x() * delta.x() + delta.y() * delta.y() < minimumDistanceSquared) {
      return true;
    }
  }

  return false;
}

QVector<MarkerCandidate> markerCandidatesFromDensity(const QVector<unsigned char> &mask,
                                                     int width,
                                                     int height,
                                                     const QRect &bounds,
                                                     const Transformation &transformation,
                                                     double yMinimum,
                                                     double yMaximum,
                                                     const QVector<MarkerCandidate> &existingCandidates)
{
  QVector<MarkerCandidate> candidates;
  const int radius = 5;
  const int diameter = radius * 2 + 1;
  const int minimumForeground = 10;

  struct DensityPeak
  {
    QPoint center;
    int foreground;
  };
  QVector<DensityPeak> peaks;

  for (int y = bounds.top() + radius; y <= bounds.bottom() - radius; y += 2) {
    for (int x = bounds.left() + radius; x <= bounds.right() - radius; x += 2) {
      int foreground = 0;
      for (int yy = y - radius; yy <= y + radius; ++yy) {
        for (int xx = x - radius; xx <= x + radius; ++xx) {
          foreground += mask [indexFor(xx, yy, width)];
        }
      }

      if (foreground < minimumForeground) {
        continue;
      }

      bool localMaximum = true;
      for (int yy = y - 2; yy <= y + 2 && localMaximum; yy += 2) {
        for (int xx = x - 2; xx <= x + 2; xx += 2) {
          if (xx == x && yy == y) {
            continue;
          }

          int neighborForeground = 0;
          for (int wy = yy - radius; wy <= yy + radius; ++wy) {
            for (int wx = xx - radius; wx <= xx + radius; ++wx) {
              if (wx < bounds.left() || wx > bounds.right() ||
                  wy < bounds.top() || wy > bounds.bottom()) {
                continue;
              }
              neighborForeground += mask [indexFor(wx, wy, width)];
            }
          }

          if (neighborForeground > foreground) {
            localMaximum = false;
            break;
          }
        }
      }

      if (localMaximum) {
        DensityPeak peak;
        peak.center = QPoint(x, y);
        peak.foreground = foreground;
        peaks << peak;
      }
    }
  }

  std::sort(peaks.begin(),
            peaks.end(),
            [] (const DensityPeak &left, const DensityPeak &right) {
              return left.foreground > right.foreground;
            });

  for (int index = 0; index < peaks.count(); ++index) {
    if (candidates.count() >= 500) {
      break;
    }

    const QPoint center = clampPointToYRange(peaks.at(index).center,
                                             transformation,
                                             yMinimum,
                                             yMaximum);
    if (!bounds.contains(center) ||
        pointIsNearExistingCandidate(center, existingCandidates, CURVE_POINT_MIN_SPACING) ||
        pointIsNearExistingCandidate(center, candidates, CURVE_POINT_MIN_SPACING)) {
      continue;
    }

    Component component;
    component.area = peaks.at(index).foreground;
    component.sumX = static_cast<long long> (center.x()) * peaks.at(index).foreground;
    component.sumY = static_cast<long long> (center.y()) * peaks.at(index).foreground;
    component.minX = qMax(bounds.left(), center.x() - radius);
    component.maxX = qMin(bounds.right(), center.x() + radius);
    component.minY = qMax(bounds.top(), center.y() - radius);
    component.maxY = qMin(bounds.bottom(), center.y() + radius);

    MarkerCandidate candidate;
    candidate.center = center;
    candidate.bounds = QRect(QPoint(component.minX, component.minY),
                             QPoint(component.maxX, component.maxY));
    candidate.area = peaks.at(index).foreground;
    candidate.size = diameter;
    candidate.patch = extractNormalizedPatch(mask,
                                             width,
                                             height,
                                             component);
    candidate.features = markerFeatures(candidate);
    candidates << candidate;
  }

  return candidates;
}

void updateClusterCentroid(MarkerCluster &cluster,
                           const QVector<MarkerCandidate> &candidates)
{
  if (cluster.candidateIndexes.isEmpty()) {
    cluster.centroid.clear();
    return;
  }

  const int featureCount = candidates.at(cluster.candidateIndexes.first()).features.count();
  cluster.centroid = QVector<double>(featureCount, 0.0);

  for (int clusterIndex = 0; clusterIndex < cluster.candidateIndexes.count(); ++clusterIndex) {
    const MarkerCandidate &candidate = candidates.at(cluster.candidateIndexes.at(clusterIndex));
    for (int featureIndex = 0; featureIndex < featureCount; ++featureIndex) {
      cluster.centroid [featureIndex] += candidate.features.at(featureIndex);
    }
  }

  for (int featureIndex = 0; featureIndex < featureCount; ++featureIndex) {
    cluster.centroid [featureIndex] /= cluster.candidateIndexes.count();
  }
}

QVector<MarkerCluster> clusterMarkerCandidates(const QVector<MarkerCandidate> &candidates)
{
  QVector<MarkerCluster> clusters;

  for (int candidateIndex = 0; candidateIndex < candidates.count(); ++candidateIndex) {
    const MarkerCandidate &candidate = candidates.at(candidateIndex);
    int bestCluster = -1;
    double bestDistance = std::numeric_limits<double>::max();

    for (int clusterIndex = 0; clusterIndex < clusters.count(); ++clusterIndex) {
      const double shapeDistance = featureDistance(candidate.features,
                                                   clusters [clusterIndex].centroid);
      const MarkerCandidate &representative = candidates.at(clusters [clusterIndex].candidateIndexes.first());
      const double sizeDistance = std::abs(candidate.size - representative.size) /
                                  qMax(1.0, qMax(candidate.size, representative.size));
      const double distance = shapeDistance + sizeDistance * 0.08;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestCluster = clusterIndex;
      }
    }

    if (bestCluster >= 0 && bestDistance <= MARKER_CLUSTER_DISTANCE_THRESHOLD) {
      clusters [bestCluster].candidateIndexes << candidateIndex;
      updateClusterCentroid(clusters [bestCluster], candidates);
    } else {
      MarkerCluster cluster;
      cluster.candidateIndexes << candidateIndex;
      cluster.centroid = candidate.features;
      cluster.meanDistance = 0.0;
      cluster.sizeVariation = 0.0;
      clusters << cluster;
    }
  }

  for (int clusterIndex = 0; clusterIndex < clusters.count(); ++clusterIndex) {
    MarkerCluster &cluster = clusters [clusterIndex];
    double distanceTotal = 0.0;
    double sizeTotal = 0.0;
    for (int index = 0; index < cluster.candidateIndexes.count(); ++index) {
      const MarkerCandidate &candidate = candidates.at(cluster.candidateIndexes.at(index));
      distanceTotal += featureDistance(candidate.features, cluster.centroid);
      sizeTotal += candidate.size;
    }

    const double meanSize = sizeTotal / qMax(1, cluster.candidateIndexes.count());
    double sizeVariance = 0.0;
    for (int index = 0; index < cluster.candidateIndexes.count(); ++index) {
      const MarkerCandidate &candidate = candidates.at(cluster.candidateIndexes.at(index));
      const double delta = candidate.size - meanSize;
      sizeVariance += delta * delta;
    }

    cluster.meanDistance = distanceTotal / qMax(1, cluster.candidateIndexes.count());
    cluster.sizeVariation = std::sqrt(sizeVariance / qMax(1, cluster.candidateIndexes.count())) /
                            qMax(1.0, meanSize);
  }

  return clusters;
}

QList<AutoCurveGroup> groupsFromClusters(const QVector<MarkerCluster> &clusters,
                                         const QVector<MarkerCandidate> &candidates)
{
  QList<AutoCurveGroup> groups;
  int bestCount = 0;
  for (int index = 0; index < clusters.count(); ++index) {
    bestCount = qMax(bestCount, clusters.at(index).candidateIndexes.count());
  }

  const int minimumGroupSize = bestCount >= 4 ? qMax(2, bestCount / 5) : 2;

  for (int clusterIndex = 0; clusterIndex < clusters.count(); ++clusterIndex) {
    const MarkerCluster &cluster = clusters.at(clusterIndex);
    if (cluster.candidateIndexes.count() < minimumGroupSize) {
      continue;
    }

    AutoCurveGroup group;
    for (int index = 0; index < cluster.candidateIndexes.count(); ++index) {
      addCurvePointIfSeparated(group.points,
                               candidates.at(cluster.candidateIndexes.at(index)).center);
    }

    if (group.points.count() < minimumGroupSize) {
      continue;
    }

    std::sort(group.points.begin(),
              group.points.end(),
              [] (const QPoint &left, const QPoint &right) {
                if (left.x() == right.x()) {
                  return left.y() < right.y();
                }
                return left.x() < right.x();
              });

    group.confidence = static_cast<double> (group.points.count()) /
                       (1.0 + cluster.meanDistance * 10.0 + cluster.sizeVariation * 3.0);
    groups << group;
  }

  std::sort(groups.begin(),
            groups.end(),
            [] (const AutoCurveGroup &left, const AutoCurveGroup &right) {
              if (left.points.count() == right.points.count()) {
                return left.confidence > right.confidence;
              }
              return left.points.count() > right.points.count();
            });

  for (int index = 0; index < groups.count(); ++index) {
    groups [index].name = QObject::tr("Marker Group %1").arg(index + 1);
  }

  return groups;
}

QPoint clampPointToYRange(const QPointF &pointScreen,
                          const Transformation &transformation,
                          double yMinimum,
                          double yMaximum)
{
  QPointF graph;
  transformation.transformScreenToRawGraph(pointScreen, graph);
  graph.setY(qBound(yMinimum, graph.y(), yMaximum));

  QPointF adjustedScreen;
  transformation.transformRawGraphToScreen(graph, adjustedScreen);
  return QPoint(qRound(adjustedScreen.x()), qRound(adjustedScreen.y()));
}

void addCurvePointIfSeparated(QList<QPoint> &points,
                              const QPoint &point)
{
  for (int index = 0; index < points.count(); ++index) {
    const QPoint existing = points.at(index);
    const int dx = existing.x() - point.x();
    const int dy = existing.y() - point.y();
    if (dx * dx + dy * dy < CURVE_POINT_MIN_SPACING * CURVE_POINT_MIN_SPACING) {
      return;
    }
  }

  points << point;
}

} // namespace

AutoCurveGroup::AutoCurveGroup() :
  confidence (0.0)
{
}

AutoCurveResult::AutoCurveResult()
{
}

AutoAxisResult::AutoAxisResult() :
  success (false),
  rotated (false),
  rotationDegrees (0.0)
{
}

AutoAxisResult AutoDigitize::detectAxes(const QImage &image,
                                        double yMaximum)
{
  AutoAxisResult result;
  if (image.isNull()) {
    result.message = QObject::tr("No image is loaded.");
    return result;
  }

  QImage working = image.convertToFormat(QImage::Format_ARGB32);
  const double detectedAngle = estimateHorizontalAngleDegrees(working);
  if (std::abs(detectedAngle) >= AXIS_ROTATION_THRESHOLD_DEGREES &&
      std::abs(detectedAngle) <= MAX_AXIS_ROTATION_DEGREES) {
    working = rotateImage(working, -detectedAngle);
    result.rotated = true;
    result.rotationDegrees = -detectedAngle;
  }

  const QVector<unsigned char> mask = foregroundMask(working);
  Run xAxisRun;
  if (!detectHorizontalAxis(mask, working.width(), working.height(), xAxisRun)) {
    result.message = QObject::tr("Could not find a clear horizontal x-axis line.");
    return result;
  }

  Run yAxisRun;
  if (!detectVerticalAxis(mask, working.width(), working.height(), xAxisRun, yAxisRun)) {
    yAxisRun.start = qMax(0, xAxisRun.rowOrColumn - working.height() / 2);
    yAxisRun.end = xAxisRun.rowOrColumn;
    yAxisRun.rowOrColumn = xAxisRun.start;
    yAxisRun.length = yAxisRun.end - yAxisRun.start + 1;
  }

  const int yAxisX = qBound(0, yAxisRun.rowOrColumn, working.width() - 1);
  const int xAxisY = qBound(0, xAxisRun.rowOrColumn, working.height() - 1);
  const int xAxisRightMinimum = qMin(yAxisX + 1, working.width() - 1);
  const int xAxisRight = qBound(xAxisRightMinimum, xAxisRun.end, working.width() - 1);
  const int yAxisTopMaximum = qMax(0, xAxisY - 1);
  const int yAxisTop = qBound(0, yAxisRun.start, yAxisTopMaximum);
  if (xAxisRight <= yAxisX || yAxisTop >= xAxisY) {
    result.message = QObject::tr("Detected axes are too small to calibrate.");
    return result;
  }

  const double xMaximum = qMax(2.0, static_cast<double> (xAxisRight - yAxisX + 1));

  result.pointsScreen << QPointF(yAxisX, xAxisY)
                      << QPointF(xAxisRight, xAxisY)
                      << QPointF(yAxisX, yAxisTop);
  result.pointsGraph << QPointF(1.0, 0.0)
                     << QPointF(xMaximum, 0.0)
                     << QPointF(1.0, yMaximum);
  result.image = working;
  result.success = true;
  result.message = QObject::tr("Auto axis created three calibration points.");

  return result;
}

QList<QPoint> AutoDigitize::detectCurvePoints(const QImage &image,
                                              const QRect &plotRect,
                                              const Transformation &transformation,
                                              double yMinimum,
                                              double yMaximum)
{
  const AutoCurveResult result = detectCurvePointGroups(image,
                                                        plotRect,
                                                        transformation,
                                                        yMinimum,
                                                        yMaximum);
  QList<QPoint> points;
  if (!result.groups.isEmpty()) {
    points = result.groups.first().points;
  }

  return points;
}

AutoCurveResult AutoDigitize::detectCurvePointGroups(const QImage &image,
                                                     const QRect &plotRect,
                                                     const Transformation &transformation,
                                                     double yMinimum,
                                                     double yMaximum)
{
  AutoCurveResult result;
  if (image.isNull() || plotRect.isEmpty() || !transformation.transformIsDefined()) {
    result.message = QObject::tr("No calibrated image area is available for Auto Curve.");
    return result;
  }

  const QImage working = image.convertToFormat(QImage::Format_ARGB32);
  const int width = working.width();
  const int height = working.height();
  const QRect imageRect(0, 0, width, height);
  const QRect bounds = plotRect.adjusted(4, 4, -4, -4).intersected(imageRect);
  if (bounds.isEmpty()) {
    result.message = QObject::tr("No calibrated image area is available for Auto Curve.");
    return result;
  }

  const QVector<unsigned char> mask = foregroundMask(working);
  const QVector<unsigned char> lineSuppressedMask = suppressLongRuns(mask,
                                                                     width,
                                                                     height,
                                                                     bounds);
  const QVector<unsigned char> denseMask = denseForegroundMask(lineSuppressedMask,
                                                              width,
                                                              height,
                                                              bounds);
  QVector<MarkerCandidate> candidates = markerCandidatesFromComponents(denseMask,
                                                                       width,
                                                                       height,
                                                                       bounds,
                                                                       transformation,
                                                                       yMinimum,
                                                                       yMaximum);
  const QVector<MarkerCandidate> densityCandidates = markerCandidatesFromDensity(lineSuppressedMask,
                                                                                width,
                                                                                height,
                                                                                bounds,
                                                                                transformation,
                                                                                yMinimum,
                                                                                yMaximum,
                                                                                candidates);
  candidates += densityCandidates;
  const QVector<MarkerCluster> clusters = clusterMarkerCandidates(candidates);
  result.groups = groupsFromClusters(clusters,
                                     candidates);

  if (result.groups.isEmpty()) {
    result.message = QObject::tr("No repeated marker groups were detected inside the calibrated plot area.");
  } else {
    int pointCount = 0;
    for (int index = 0; index < result.groups.count(); ++index) {
      pointCount += result.groups.at(index).points.count();
    }
    result.message = QObject::tr("Detected %1 marker groups with %2 candidate points.")
                     .arg(result.groups.count())
                     .arg(pointCount);
  }

  return result;
}
