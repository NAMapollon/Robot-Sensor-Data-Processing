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

#****************************Here is another code for 3D reconstruction******************************
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

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
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

Q2 = np.float32([[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, -1, 0],
                 [0, 0, 0, 1]])
points_3D = cv2.reprojectImageTo3D(norm_disp8, Q2)
colors = cv2.cvtColor(imgLU, cv2.COLOR_BGR2RGB)
mask_map = norm_disp8 > norm_disp8.min()

output_points = points_3D[mask_map]
output_colors = colors[mask_map]
output_file = 'reconstructed.ply'
create_output(output_points, output_colors, output_file)
# **************************************************************************************************
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
