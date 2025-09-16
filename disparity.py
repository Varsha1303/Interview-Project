import os
import cv2
import numpy as np

# ------------------ preprocessing code ------------------
def preprocess_image(img, use_clahe=True, blur_ksize=3, sharpen=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Using CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    else:
        gray = cv2.equalizeHist(gray)

    # Small Gaussian blur
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Edge sharpening
    if sharpen:
        lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        gray = cv2.addWeighted(gray, 1.0, lap, -0.3, 0)

    # Normalize to [0,255]
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray

# ------------------ stereo matching ------------------
def compute_disparities(left_img, right_img, ndisp=256, block_size=5):
    P1 = 8 * 3 * block_size * block_size
    P2 = 32 * 3 * block_size * block_size

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=ndisp,
        blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=2,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=16,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    disp_left = left_matcher.compute(left_img, right_img).astype(np.float32) / 16.0
    disp_right = right_matcher.compute(right_img, left_img).astype(np.float32) / 16.0

    return disp_left, disp_right

# ------------------ post-processing with WLS ------------------
def postprocess_disparity(disp_left, disp_right, left_img):
    # Using WLS filter
    wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)

    disp_filtered = wls.filter(disp_left, left_img, disparity_map_right=disp_right)

    # Normalize
    disp_filtered[disp_filtered < 0] = 0
    disp8 = cv2.normalize(disp_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Small median blur
    disp8 = cv2.medianBlur(disp8, 3)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    disp8 = cv2.morphologyEx(disp8, cv2.MORPH_CLOSE, kernel)

    return disp8

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "data", "Adirondack-perfect", "Adirondack-perfect")
    left_path = os.path.join(base, "im0.png")
    right_path = os.path.join(base, "im1.png")

    left_color = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_color = cv2.imread(right_path, cv2.IMREAD_COLOR)

    # Step 1: Preprocess
    left_gray = preprocess_image(left_color, use_clahe=True)
    right_gray = preprocess_image(right_color, use_clahe=True)

    # Step 2: Compute disparities using (left + right)
    disp_left, disp_right = compute_disparities(left_gray, right_gray, ndisp=256, block_size=5)

    # Step 3: Post-process with WLS, median and morphology
    disp_final = postprocess_disparity(disp_left, disp_right, left_gray)

    # Step 4: Generating Colored disparity map
    disp_color = cv2.applyColorMap(disp_final, cv2.COLORMAP_JET)

    # Save the outputs
    cv2.imwrite("disp_gray_1.png", disp_final)
    cv2.imwrite("disp_color_1.png", disp_color)
