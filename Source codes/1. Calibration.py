import numpy as np
import cv2
import glob
import open3d as o3d

supervise_mode = True

# Your information here
name = 'Nam seung ha'
student_id = '2017103719'

if supervise_mode:
    print('name:%s id:%s'%(name, student_id))

# ====================================================================================
# Camera calibration
# ====================================================================================
# Set directory path (images capturing check pattern)
# Example) calibration_dir_path = 'calibration/*.png'
calibration_dir_path = 'calibration images folder/*.png'
calibration_images = glob.glob(calibration_dir_path)

# intrinsic parameters and distortion coefficient
# With these parameter, you can get undistorted image and new intrinsic parameter of them (K_undist)
K = np.array([], dtype=np.float32)
dist = np.array([], dtype=np.float32)
# new matrix for undistorted intrinsic parameter
K_undist = np.array([], dtype=np.float32)

# Your code here
# Goals
# 1. Get camera intrinsic parameters from your captured images K, dist, K_undist
# 2. Try to get undistorted images by warping captured images using K_undist
# reference: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
# ****************************** Your code here (M-1) ******************************
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)

objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calibration images folder/calib1_1.png')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (400, 500)) # Resize for better result.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2] # h for height, w for width.
    K_undist, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

if supervise_mode:
    print('1-1. Calibration: K matrix')
    print(K)
    print('1-2. Calibration: distortion coefficients')
    print(dist)
    print('1-3. Calibration: Undistorted K matrix')
    print(K_undist)
    for fname in calibration_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_undist = cv2.undistort(gray, K, dist, None, K_undist)
        cv2.imshow('undistorted image', img_undist)
        cv2.waitKey(0)
# ====================================================================================
# load stereo images (Left and Right)
# ====================================================================================
#  set your left and right images
# Example
# imgL = cv2.imread('stereo/left.png')
# imgR = cv2.imread('stereo/right.png')
imgR = cv2.imread('stereo image/left.png')
imgL = cv2.imread('stereo image/right.png')
# convert to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# Convert to undistorted images
# imgLU: undistorted image of imgL
# imgRU: undistorted image of imgR
# grayLU: undistorted image of grayL
# grayRU: undistorted image of grayR
imgLU  = np.array([])
imgRU  = np.array([])
grayLU = np.array([])
grayRU = np.array([])
