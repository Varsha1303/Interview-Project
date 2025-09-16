import os
import cv2
import numpy as np

def parse_calib_file(calib_file):
    f, cx, cy, baseline = None, None, None, None
    with open(calib_file, "r") as f_calib:
        for line in f_calib:
            if line.startswith("cam0"):
                values = line.split("=")[1].strip(" []\n").replace(";", " ").split()
                values = [float(v.strip(",")) for v in values]
                fx, fy, cx, cy = values[0], values[4], values[2], values[5]
                f = fx 
            elif line.startswith("baseline"):
                baseline = float(line.split("=")[1])
    if None in [f, cx, cy, baseline]:
        raise ValueError("Could not parse calibration file correctly.")
    return f, cx, cy, baseline

def disparity_to_pointcloud(disp, left_img, calib_file, out_file="pointcloud.ply"):
    f, cx, cy, B = parse_calib_file(calib_file)

    h, w = disp.shape
    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, f],
        [0, 0, -1.0/B, 0]
    ])

    points_3D = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()

    output_points = points_3D[mask]
    output_colors = colors[mask]

    write_ply(out_file, output_points, output_colors)
    print(f"[INFO] Point cloud saved to {out_file}")

def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "data", "Adirondack-perfect", "Adirondack-perfect")
    left_path = os.path.join(base, "im0.png")
    calib_path = os.path.join(base, "calib.txt")

    left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
    disp = cv2.imread("disp_gray_1.png", cv2.IMREAD_GRAYSCALE)  # already computed disparity

    disparity_to_pointcloud(disp, left_img, calib_path, out_file="Adirondack_pointcloud.ply")
