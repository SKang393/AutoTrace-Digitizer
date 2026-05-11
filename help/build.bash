# This bash script creates .qch and .qhc output files. Optional argument is target directory.
# Since bash has different locations depending on the operating system, no #! line is included.
#
# Usage: .../build_qt5_12_0.bash [destdir]
# where:
#   [destdir]   Optional destination directory relative to the directory containing this script.
#               Default is ../bin/documentation
#
# This version is for qt 5.12.0 or newer, for which qcollectiongenerator is deprecated in favor of
# qhelpgenerator. Earlier versions of qcollectiongenerator fail since qch file is NOT generated
QMAKE=${QMAKE:-}
if [ -z "$QMAKE" ]; then
    if command -v qmake6 >/dev/null 2>&1; then
        QMAKE=qmake6
    elif command -v qmake >/dev/null 2>&1; then
        QMAKE=qmake
    else
        echo "Warning: qmake was not found. Skipping optional searchable help generation."
        exit 0
    fi
fi

VER=`$QMAKE -query QT_VERSION`
MAJOR=$(echo $VER | cut -f1 -d. )
MINOR=$(echo $VER | cut -f2 -d. )
SUBMINOR=$(echo $VER | cut -f3 -d. )
MAJORMIN=5
MINORMIN=12
SUBMINORMIN=0
SUCCESS=1
if (( MAJOR < MAJORMIN )); then
    SUCCESS=0
elif (( MAJOR == MAJORMIN )); then
    if (( MINOR < MINORMIN )); then
	SUCCESS=0
    elif (( MINOR == MINORMIN )); then
	if (( SUBMINOR < SUBMINORMIN )); then
	    SUCCESS=0
	fi
    fi
fi    
if (( SUCCESS == 0 )); then
    echo "Qt version must be $MAJORMIN.$MINORMIN.$SUBMINORMIN or newer. Use another build script"
    exit
fi    
# Run this script in the 'help' subdirectory to prevent confusion about $1 argument being relative
# script directory or current working directory. Flathub build script likes the 'cd' command in
# this script rather than outside
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
WORKPATH=`pwd`
WORKDIR=`basename "$WORKPATH"`
if [[ $WORKDIR != "help" ]]; then
    # Changing to script directory
    cd "$SCRIPTDIR"
fi

if [ $# -eq 0 ]; then
    DESTDIR=../bin/documentation
else
    DESTDIR=$1
fi

# Specifying ../bin/documentation directory here creates files in this directory
# AND ../bin/documentation (with different sizes) so we build into the local directory.
# Outputs:
#   engauge.qch 3MB binary file
#   engauge.qhc 32K SQLite file
HELPGENERATOR=${HELPGENERATOR:-}
if [ -z "$HELPGENERATOR" ]; then
    if command -v qhelpgenerator >/dev/null 2>&1; then
        HELPGENERATOR=qhelpgenerator
    elif command -v qhelpgenerator-qt6 >/dev/null 2>&1; then
        HELPGENERATOR=qhelpgenerator-qt6
    elif command -v qhelpgenerator6 >/dev/null 2>&1; then
        HELPGENERATOR=qhelpgenerator6
    elif [ -n "`$QMAKE -query QT_INSTALL_BINS 2>/dev/null`" ] && [ -x "`$QMAKE -query QT_INSTALL_BINS`/qhelpgenerator" ]; then
        HELPGENERATOR="`$QMAKE -query QT_INSTALL_BINS`/qhelpgenerator"
    elif [ -n "`$QMAKE -query QT_INSTALL_BINS 2>/dev/null`" ] && [ -x "`$QMAKE -query QT_INSTALL_BINS`/qhelpgenerator.exe" ]; then
        HELPGENERATOR="`$QMAKE -query QT_INSTALL_BINS`/qhelpgenerator.exe"
    else
        echo "Warning: qhelpgenerator was not found. Skipping optional searchable help generation."
        exit 0
    fi
fi

"$HELPGENERATOR" engauge.qhp

# Move to target directory which is relative to the help subdirectory. OSX does not use qhc file
mkdir -p "$DESTDIR"
mv engauge.qch "$DESTDIR"
if [ -f engauge.qhc ]; then
    mv engauge.qhc "$DESTDIR"
fi

