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
# stereo matching (Dense matching)
# ====================================================================================
# Goals
#  1. Get disparity map (8 bit unsigned)
#  Note. The output of disparity function (StereoBM, etc.) is 16-bit
# reference: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_depthmap.html
disp8 = np.array([], np.uint8)
# ****************************** Your code here (M-3) *******************************
stereo = cv2.StereoBM_create(numDisparities = 32, blockSize = 19) # Use various numDisparities and blockSize.
disp8 = stereo.compute(grayLU, grayRU)
norm_disp8 = cv2.normalize(disp8, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F) # Normalization image for better result.
#************************************************************************************
if supervise_mode:
    imgLU[disp8 < 1, :] = 0
    cv2.imshow('disparity', norm_disp8)
    cv2.imshow('Left Post-processing', imgLU)
    cv2.waitKey(0)
