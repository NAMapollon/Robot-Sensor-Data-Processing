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
# Visualization
# ====================================================================================
# In advance, you should install open3D (open3d.org)
# pip install open3d
#
pcd = o3d.geometry.PointCloud()
#  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
#  pc_color: array(Nx3), each row composed with R G,B in the rage of 0~1
pc_points = np.array([], np.float32)
pc_color = np.array([], np.float32)
#3D reconstruction
#Concatenate pc_points and pc_color
#****************************** Your code here (M-4) ******************************
h, w = grayLU.shape[:2]
f = 0.8 * w # f means 'focal length'.
Q = np.float32([[1, 0, 0, -0.5 * w],
               [0, -1, 0, 0.5 * h],
               [0, 0, 0, -f],
               [0, 0, 1, 0]]) # Q is a matrix for disparity-to-depth mapping.
pc_points = cv2.reprojectImageTo3D(norm_disp8, Q) # Reproject 2D disparity image to 3D.
pc_color = cv2.cvtColor(grayLU, cv2.COLOR_BGR2RGB)
mask = norm_disp8 > norm_disp8.min()
mask = norm_disp8 < norm_disp8.max()
out_points = pc_points[mask]
out_colors = pc_color[mask]

# add position and color to point cloud
pcd.points = o3d.utility.Vector3dVector(out_points)
pcd.colors = o3d.utility.Vector3dVector(out_colors)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

cv2.destroyAllWindows()
#  end of code
