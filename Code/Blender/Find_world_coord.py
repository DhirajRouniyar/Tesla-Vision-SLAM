import numpy as np
# import scipy.io
import json



# Load the .mat file
# K1 = np.load('./camera_data/cam_mtx.npy')
# K2 = np.load('./camera_data/newcam_mtx.npy')
K3 = np.array([[1.594657424424391e+03,             0,  6.552961052379301e+02],
            [            0, 1.607694179766480e+03, 4.143627123354900e+02],
            [            0,             0,             1]]).reshape(3,3)

# print("K3:", K3)


def pixel_to_world(u, v, scaling, K, R, T):

    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)

    uv_1 = np.array([[u, v, 1]], dtype=np.float32)
    uv_1 = uv_1.T
    scale_uv_1 = uv_1 * scaling
    xyz = np.dot(K_inv, scale_uv_1)
    xyz = xyz - T
    
    XYZ = np.dot(R_inv, xyz)
    XYZ = np.around(XYZ, decimals=4)
    return XYZ.flatten()


    # world_coords = R.T @ np.linalg.inv(K) @ np.array([u, v, 1]) * scaling
    # x = world_coords[0]
    # y = world_coords[1] 
    # z = world_coords[2]
    # return x,y,z
# Camera Intrinsic parameters: #Trial
# K = np.array([[531.122155322710,             0,  407.192550839899],
#             [            0, 531.541737503901, 313.308715048366],
#             [            0,             0,             1]]).reshape(3,3)


def main(u, v, depth):

    x_angle = np.deg2rad(0)
    y_angle = np.deg2rad(0)
    z_angle = np.deg2rad(0)

    # Calculate rotation matrices for individual axes
    Rx = np.array([[1, 0, 0],
                [0, np.cos(x_angle), -np.sin(x_angle)],
                [0, np.sin(x_angle), np.cos(x_angle)]])

    Ry = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                [0, 1, 0],
                [-np.sin(y_angle), 0, np.cos(y_angle)]])

    Rz = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                [np.sin(z_angle), np.cos(z_angle), 0],
                [0, 0, 1]])

    # Combine rotation matrices to obtain the overall rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))

    T = np.array([0, 0, 0])  # Translation vector
    T_transpose = (T.T).reshape(3, 1)
    # Example usage
    # u = (top_left[0] + bottom_right[0]) / 2
    # v = (top_left[1] + bottom_right[1]) / 2
    # u, v = 100, 200  # Image pixel coordinates
    scaling = depth  # Depth value in meters
    
    X, Y, Z = pixel_to_world(u, v, scaling, K3, R, T_transpose)  #updated
    # X, Y, Z = pixel_to_world(u, v, scaling, K3, R)
    # print('World coordinates: {X, Y, Z}', X, Y, Z) 
    return X, Y, Z
    

    # print("World coordinates:", X, Y, Z)
# main()