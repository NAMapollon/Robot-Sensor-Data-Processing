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

pcd = o3d.geometry.PointCloud()

#  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
#  pc_color: array(Nx3), each row composed with R G,B in the rage of 0~1
pc_points = np.array([], np.float32) 
pc_color = np.array([], np.float32) 

# 3D reconstruction
# Concatenate pc_points and pc_color
# ****************************** Your code here (M-4) ******************************
# Get intrinsic parameter
# Focal length
fx = K_undist[0][0]
fy = K_undist[1][1]
# Principal point
U0 = K_undist[0][2]
V0 = K_undist[1][2]

# RGB to BGR for pc_points
imgLU = cv2.cvtColor(imgLU, cv2.COLOR_RGB2BGR)

# depth = inverse of disparity
depth = 255 - disp8

for v in range(h):
    for u in range(w):
        if(disp8[v][u] > 0):
            # pc_points
            x = (u - U0) * depth[v][u] / fx
            y = (v - V0) * depth[v][u] / fy
            z = depth[v][u]
            pc_points = np.append(pc_points, np.array(np.float32(([x, y, z]))))
            pc_points = np.reshape(pc_points, (-1, 3))
            # pc_colors
            pc_color = np.append(pc_color, np.array(np.float32(imgLU[v][u] / 255)))
            pc_color = np.reshape(pc_color, (-1, 3))

# **********************************************************************************
#  add position and color to point cloud
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_color)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
cv2.destroyAllWindows()
#  end of code
