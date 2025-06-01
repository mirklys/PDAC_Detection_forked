#!/bin/bash

MAIN_OUTPUT_DIR="/home/bobby/repos/PDAC_Detection_forked/workspace/test/labelsTs/"

COMBINED_DIR="$MAIN_OUTPUT_DIR/tversky_predicted"

COMBINED_MAPS_DIR="$COMBINED_DIR/pdac-detection-map"

mkdir -p "$COMBINED_MAPS_DIR"

echo "Combining results from: $MAIN_OUTPUT_DIR"
echo "Destination: $COMBINED_DIR"
echo "==========================================="


echo "Moving all .nii.gz detection maps..."

find "$MAIN_OUTPUT_DIR"/batch_*/pdac-detection-map -type f -name "*.nii.gz" -exec mv -t "$COMBINED_MAPS_DIR" {} +

echo "All detection maps have been moved to $COMBINED_MAPS_DIR"
echo "-------------------------------------------"


echo "Merging all pdac-likelihood.json files..."

jq -s 'add' "$MAIN_OUTPUT_DIR"/batch_*/pdac-likelihood.json > "$COMBINED_DIR/pdac-likelihood.json"

echo "All likelihood scores have been merged into $COMBINED_DIR/pdac-likelihood.json"
echo "-------------------------------------------"