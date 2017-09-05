#! /bin/bash

TARGET_PDF=$1

echo "Target:" $TARGET_PDF

TARGET_DIR=${TARGET_PDF%/*/*}

echo "Target dir:" $TARGET_DIR

convert -delay 33 -loop 0 $TARGET_PDF $TARGET_DIR/all_dyn.gif
