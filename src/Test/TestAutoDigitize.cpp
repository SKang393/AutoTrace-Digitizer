#include "AutoDigitize.h"
#include "DocumentModelCoords.h"
#include "DocumentModelGeneral.h"
#include "Logger.h"
#include "MainWindowModel.h"
#include "Test/TestAutoDigitize.h"
#include "Transformation.h"
#include <QImage>
#include <QPainter>
#include <QPolygon>
#include <QtTest/QtTest>

QTEST_MAIN (TestAutoDigitize)

namespace {

QList<QPoint> allPointsFromGroups(const QList<AutoCurveGroup> &groups)
{
  QList<QPoint> points;
  for (int groupIndex = 0; groupIndex < groups.count(); ++groupIndex) {
    for (int pointIndex = 0; pointIndex < groups.at(groupIndex).points.count(); ++pointIndex) {
      points << groups.at(groupIndex).points.at(pointIndex);
    }
  }

  return points;
}

QString pointText(const QPoint &point)
{
  return QString ("(%1,%2)").arg(point.x()).arg(point.y());
}

} // namespace

TestAutoDigitize::TestAutoDigitize(QObject *parent) :
  QObject(parent)
{
}

void TestAutoDigitize::cleanupTestCase ()
{
}

void TestAutoDigitize::initTestCase ()
{
  initializeLogging ("engauge_test",
                     "engauge_test.log",
                     false);
}

bool TestAutoDigitize::containsPointNear(const QList<QPoint> &points,
                                         const QPoint &expected,
                                         int tolerance) const
{
  const int toleranceSquared = tolerance * tolerance;
  for (int index = 0; index < points.count(); ++index) {
    const QPoint delta = points.at(index) - expected;
    if (delta.x() * delta.x() + delta.y() * delta.y() <= toleranceSquared) {
      return true;
    }
  }

  return false;
}

QImage TestAutoDigitize::regressionImage() const
{
  QImage image(850, 270, QImage::Format_ARGB32);
  image.fill(Qt::white);

  QPainter painter(&image);
  painter.setRenderHint(QPainter::Antialiasing, false);

  painter.setPen(QPen(Qt::black, 1));
  painter.drawLine(55, 239, 789, 239);
  painter.drawLine(55, 31, 55, 239);

  QPen gridPen(QColor(190, 190, 190), 1, Qt::DashLine);
  painter.setPen(gridPen);
  painter.drawLine(55, 135, 789, 135);
  painter.drawLine(420, 31, 420, 239);

  painter.setPen(QPen(Qt::black, 1, Qt::DotLine));
  painter.drawLine(214, 31, 214, 239);
  painter.setPen(QPen(Qt::black, 2));
  painter.drawLine(779, 31, 779, 239);

  painter.setPen(QPen(Qt::black, 1));
  painter.setBrush(Qt::NoBrush);
  painter.drawRect(702, 204, 70, 24);
  painter.drawText(QRect(704, 204, 66, 24),
                   Qt::AlignCenter,
                   "P1 Box");

  painter.setPen(QPen(Qt::black, 1));
  painter.setBrush(Qt::black);
  const QList<QPoint> filledCircles = {
    QPoint(96, 220),
    QPoint(150, 207),
    QPoint(318, 170),
    QPoint(475, 142)
  };
  for (int index = 0; index < filledCircles.count(); ++index) {
    painter.drawEllipse(filledCircles.at(index), 5, 5);
  }

  painter.setBrush(Qt::NoBrush);
  const QList<QPoint> openTriangles = {
    QPoint(116, 194),
    QPoint(250, 181),
    QPoint(370, 155),
    QPoint(560, 132)
  };
  for (int index = 0; index < openTriangles.count(); ++index) {
    const QPoint center = openTriangles.at(index);
    QPolygon triangle;
    triangle << QPoint(center.x(), center.y() - 6)
             << QPoint(center.x() - 6, center.y() + 6)
             << QPoint(center.x() + 6, center.y() + 6);
    painter.drawPolygon(triangle);
  }

  painter.end();
  return image;
}

Transformation TestAutoDigitize::regressionTransformation() const
{
  DocumentModelCoords modelCoords;
  modelCoords.setCoordScaleXTheta(COORD_SCALE_LINEAR);
  modelCoords.setCoordScaleYRadius(COORD_SCALE_LINEAR);
  modelCoords.setCoordsType(COORDS_TYPE_CARTESIAN);
  modelCoords.setCoordUnitsDate(COORD_UNITS_DATE_YEAR_MONTH_DAY);
  modelCoords.setCoordUnitsRadius(COORD_UNITS_NON_POLAR_THETA_NUMBER);
  modelCoords.setCoordUnitsTheta(COORD_UNITS_POLAR_THETA_DEGREES);
  modelCoords.setCoordUnitsTime(COORD_UNITS_TIME_HOUR_MINUTE_SECOND);
  modelCoords.setCoordUnitsX(COORD_UNITS_NON_POLAR_THETA_NUMBER);
  modelCoords.setCoordUnitsY(COORD_UNITS_NON_POLAR_THETA_NUMBER);
  modelCoords.setOriginRadius(0.0);

  DocumentModelGeneral modelGeneral;
  modelGeneral.setCursorSize(5);
  modelGeneral.setExtraPrecision(1);

  MainWindowModel mainWindowModel;
  QTransform matrixScreen(55, 789, 55,
                          239, 239, 31,
                          1, 1, 1);
  QTransform matrixGraph(1, 735, 1,
                         0, 0, 100,
                         1, 1, 1);

  Transformation transformation;
  transformation.setModelCoords(modelCoords,
                                modelGeneral,
                                mainWindowModel);
  transformation.updateTransformFromMatrices(matrixScreen,
                                             matrixGraph);
  return transformation;
}

void TestAutoDigitize::testSingleCaseArtifactsRejected()
{
  const AutoCurveResult result = AutoDigitize::detectCurvePointGroups(regressionImage(),
                                                                      QRect(QPoint(55, 31), QPoint(789, 239)),
                                                                      regressionTransformation(),
                                                                      0.0,
                                                                      100.0);
  QVERIFY2(result.rawCandidateCount <= 250,
           qPrintable(QString ("raw candidates=%1 message=%2").arg(result.rawCandidateCount).arg(result.message)));
  QVERIFY2(!result.groups.isEmpty(), qPrintable(result.message));

  const QList<QPoint> points = allPointsFromGroups(result.groups);
  QVERIFY2(points.count() <= 20,
           qPrintable(QString ("unexpected point count=%1").arg(points.count())));

  const QRect participantLabelRegion(702, 204, 70, 24);
  for (int index = 0; index < points.count(); ++index) {
    const QPoint point = points.at(index);
    QVERIFY2(!(point.x() >= 210 && point.x() <= 218),
             qPrintable(QString ("point on dotted phase divider %1").arg(pointText(point))));
    QVERIFY2(!(point.x() >= 778 && point.x() <= 780),
             qPrintable(QString ("point on solid phase divider %1").arg(pointText(point))));
    QVERIFY2(!participantLabelRegion.adjusted(-4, -4, 4, 4).contains(point),
             qPrintable(QString ("point in participant label/text box region %1").arg(pointText(point))));
  }
}

void TestAutoDigitize::testMarkerGroupsAreDistinctAndCentered()
{
  const AutoCurveResult result = AutoDigitize::detectCurvePointGroups(regressionImage(),
                                                                      QRect(QPoint(55, 31), QPoint(789, 239)),
                                                                      regressionTransformation(),
                                                                      0.0,
                                                                      100.0);
  QVERIFY2(result.groups.count() >= 2,
           qPrintable(QString ("groups=%1 message=%2").arg(result.groups.count()).arg(result.message)));

  const QList<QPoint> points = allPointsFromGroups(result.groups);
  QVERIFY2(containsPointNear(points, QPoint(96, 220), 3), "filled marker center was not detected");
  QVERIFY2(containsPointNear(points, QPoint(318, 170), 3), "filled marker center was not detected");
  QVERIFY2(containsPointNear(points, QPoint(250, 181), 4), "open triangle marker center was not detected");
  QVERIFY2(containsPointNear(points, QPoint(560, 132), 4), "open triangle marker center was not detected");

  for (int left = 0; left < points.count(); ++left) {
    for (int right = left + 1; right < points.count(); ++right) {
      const QPoint delta = points.at(left) - points.at(right);
      QVERIFY2(delta.x() * delta.x() + delta.y() * delta.y() >= 36,
               qPrintable(QString ("duplicate marker-sized neighborhood %1 and %2")
                          .arg(pointText(points.at(left)))
                          .arg(pointText(points.at(right)))));
    }
  }
}
