/******************************************************************************************************
 * (C) 2014 markummitchell@github.com. This file is part of Engauge Digitizer, which is released      *
 * under GNU General Public License version 2 (GPLv2) or (at your option) any later version. See file *
 * LICENSE or go to gnu.org/licenses for details. Distribution requires prior written permission.     *
 ******************************************************************************************************/

#include "DlgAbout.h"
#include "MainWindow.h"
#include <QGridLayout>
#include <QSpacerItem>
#include "Version.h"

DlgAbout::DlgAbout (MainWindow &mainWindow) :
  QMessageBox (&mainWindow),
  m_mainWindow (mainWindow)
{
  setWindowTitle (tr ("About AutoTrace Digitizer"));
  setTextFormat (Qt::RichText);

  // Do not embed single quotes in the strings below since that will interfere with the translations
  setText (QString ("<p>%1 %2 %3</p> <p>&copy; Mark Mitchell and others</p><p>%4</p><p>%5</p><p>%6</p><p>%7</p><p>%8:</p>"
                    "<ul>"
                     "<li><a href=\"https://github.com/akhuettel/engauge-digitizer\">%9</a></li>"
                    "</ul>")
           .arg (tr ("AutoTrace Digitizer"))
           .arg (tr ("Version"))
           .arg (VERSION_NUMBER)
           .arg (tr ("AutoTrace Digitizer is a modified fork of Engauge Digitizer for extracting numeric data from "
                     "images of graphs with added Auto Axis and Auto Curve tools. It is not the official upstream "
                     "Engauge Digitizer release."))
           .arg (tr ("This is free software, and you are welcome to redistribute it under "
                     "certain conditions according to the GNU General Public License Version 2,"
                     "or (at your option) any later version."))
           .arg (tr ("AutoTrace Digitizer comes with ABSOLUTELY NO WARRANTY."))
           .arg (tr ("Read the included LICENSE file for details."))
           .arg (tr ("Original Engauge Digitizer copyright and license notices are preserved."))
           .arg (tr ("Project Home Page")));

  // Calling setMinimumWidth has no effect so we insert spacer to prevent overly narrow dialog in linux.
  // Hack from https://forum.qt.io/topic/24213/qmessagebox-too-small/9
  QSpacerItem *spacer = new QSpacerItem (800,
                                         0,
                                         QSizePolicy::Minimum,
                                         QSizePolicy::Expanding);
  QGridLayout *layout = dynamic_cast<QGridLayout *> (this->layout());
  layout->addItem (spacer,
                   layout->rowCount(),
                   0,
                   1,
                   layout->columnCount());
}
