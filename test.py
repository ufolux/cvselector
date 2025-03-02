import cv2
import numpy as np

# Read source and target images
src_path = 'srcs/src.jpg'
target_path = 'targets/target-strenched.jpg'
img_src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)  # Source image in grayscale
img_target_color = cv2.imread(target_path)  # Target image in color

# Check if images are successfully loaded
if img_target_color is None:
    print("Error: Unable to load target image.")
    exit()
if img_src is None:
    print("Error: Unable to load source image.")
    exit()

# Convert the target image to grayscale for feature detection
img_target_gray = cv2.cvtColor(img_target_color, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img_src, None)
kp2, des2 = sift.detectAndCompute(img_target_gray, None)

# BFMatcher with default parameters
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# If enough good matches are found, find the matching region coordinates
if len(good) > 10:
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img_src.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Output the coordinates of the matching region
    top_left = tuple(dst[0][0])
    bottom_right = tuple(dst[2][0])
    print(f"Top-left corner of the matching region: {top_left}")
    print(f"Bottom-right corner of the matching region: {bottom_right}")

    # Draw the matching region on the original color image
    img_target_with_box = img_target_color.copy()
    cv2.polylines(img_target_with_box, [np.int32(dst)], True, (0, 255, 0), 1, cv2.LINE_AA)

    # Save the result image
    cv2.imwrite('outputs/output.jpg', img_target_with_box)
else:
    print("Not enough good matches to determine the matching region.")
