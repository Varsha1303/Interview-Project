#!/bin/bash
# Base directory
BASE="/Users/vijay/Desktop/Interview-Project/data/Adirondack-perfect/Adirondack-perfect"
LEFT_IMG="$BASE/im0.png"
CALIB_FILE="$BASE/calib.txt"
DISP_IMG="disp_gray_1.png"
PLY_OUT="Adirondack_pointcloud.ply"
echo "[1/2] Running disparity.py ..."
python3 disparity.py
if [ ! -f "$DISP_IMG" ]; then
    echo "❌ Error: disparity map $DISP_IMG not found!"
    exit 1
fi
echo "[2/2] Running point_cloud.py ..."
python3 point_cloud.py \
    --left "$LEFT_IMG" \
    --disp "$DISP_IMG" \
    --calib "$CALIB_FILE" \
    --out "$PLY_OUT"
echo "✅ Finished! Point cloud saved at $PLY_OUT"
