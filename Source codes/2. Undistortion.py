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

# undistorted images
# ****************************** Your code here (M-2) ******************************
imgLU = cv2.undistort(imgL, K, dist, None, K_undist)
imgRU = cv2.undistort(imgR, K, dist, None, K_undist)
grayLU = cv2.undistort(grayL, K, dist, None, K_undist)
grayRU = cv2.undistort(grayR, K, dist, None, K_undist)
# **********************************************************************************
if supervise_mode:
    cv2.imshow('rgb undistorted', cv2.hconcat([imgLU, imgRU]))
    cv2.imshow('gray undistorted', cv2.hconcat([grayLU, grayRU]))
    cv2.waitKey(0)
