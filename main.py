import numpy as np
import scipy.io
import cv2
import os
from scipy.io import savemat
from glob import glob


def compute_homography(pts1, pts2):
    """
    Compute the homography matrix H from points in image 1 to points in image 2.
    Args:
        pts1 (ndarray): Nx2 array of points in image 1.
        pts2 (ndarray): Nx2 array of points in image 2.
    Returns:
        H (ndarray): 3x3 homography matrix.
    """
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2, "Input points must be Nx2 arrays"
    N = pts1.shape[0]
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2 * i] = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2]
        A[2 * i + 1] = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    return H


# Step 1: Load Keypoints
data = scipy.io.loadmat('kp_gmaps.mat')
keypoints = data['kp_gmaps']
video_points = keypoints[:, :2]
map_points = keypoints[:, 2:]

# Step 2: Compute the homography matrix
H = compute_homography(video_points, map_points)

# Step 3: Save the homography matrix
savemat('output_00001.mat', {'H': H})
print("Homography matrix saved as output_00001.mat")

# Step 4: Process each frame
video_files = sorted(glob('img_*.jpg'))
for i, file in enumerate(video_files):
    frame = cv2.imread(file)
    height, width = frame.shape[:2]
    warped_frame = cv2.warpPerspective(frame, H, (width, height))
    output_file = f'output_{i:05d}.jpg'
    cv2.imwrite(output_file, warped_frame)
    savemat(f'output_{i:05d}.mat', {'H': H})

# Step 5: Process each YOLO file
yolo_files = sorted(glob('yolo_*.mat'))
for i, file in enumerate(yolo_files):
    data = scipy.io.loadmat(file)
    xyxy = data['xyxy']
    obj_ids = data['id']
    num_boxes = xyxy.shape[0]
    bbox = np.zeros_like(xyxy)
    for j in range(num_boxes):
        pt1 = np.dot(H, [xyxy[j, 0], xyxy[j, 1], 1])
        pt1 /= pt1[2]
        pt2 = np.dot(H, [xyxy[j, 2], xyxy[j, 3], 1])
        pt2 /= pt2[2]
        bbox[j] = [pt1[0], pt1[1], pt2[0], pt2[1]]
    savemat(f'yolooutput_{i:05d}.mat', {'bbox': bbox, 'id': obj_ids})