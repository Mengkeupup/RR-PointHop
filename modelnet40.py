import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import open3d as o3d

def write_ply(points, filename):
    """
    Write points (Nx3 numpy array) to a PLY file.

    Parameters:
    - points: Nx3 numpy array, where each row represents x, y, z coordinates of a point.
    - filename: String, the name of the file to write the points to.
    """
    with open(filename, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(len(points)))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        for point in points:
            ply_file.write("{} {} {}\n".format(point[0], point[1], point[2]))

def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def save_as_ply(points, filename):
    # 创建一个PointCloud对象
    pcd = o3d.geometry.PointCloud()

    # 假设points是一个Nx3的numpy数组，代表N个点的x, y, z坐标
    pcd.points = o3d.utility.Vector3dVector(points)

    # 保存点云为PLY文件
    o3d.io.write_point_cloud(filename, pcd)






def data_load(num_point=None, data_dir=None, train=True):  # , sample_size=100
    # train 一次true，保存训练数据；一次false保存test数据
    all_data_dir="C:\\Users\\29822\\Desktop\\try\\all_data"
    if not os.path.exists(data_dir):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('rm %s' % (zipfile))

    data_pth = load_dir(data_dir, name='train_files.txt' if train else 'test_files.txt')

    point_list, label_list = [], []
    for pth in data_pth:
        data_file = h5py.File(pth, 'r')
        point = data_file['data'][:]  # 假设point形状为[N, num_points, 3]
        label = data_file['label'][:]  # 假设label形状为[N, 1] 或 [N]

        # 选择前20个类别
        if num_point is not None and point.shape[1] > num_point:
            point = point[:, :num_point, :]  # 截取前 num_point 个点

        #mask = np.squeeze(label < 20)
        mask = np.squeeze((label >= 20) & (label < 40))
        point = point[mask]
        label = label[mask]

        point_list.append(point)
        label_list.append(label)

    data = np.concatenate(point_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    # 创建存储目录
    save_dir = os.path.join(all_data_dir, 'train' if train else 'test')



    for lbl in range(20,40):  # 第20到40个类别  , 40
        class_dir = os.path.join(save_dir, f"class_{lbl}")
        os.makedirs(class_dir, exist_ok=True)
        # 确保布尔索引正确应用于数据
        mask1 = label.flatten() == lbl  # 生成布尔索引，假设label形状是[N]或[N,1]
        class_data = data[mask1, ...]  # 应用布尔索引于data

       # class_data = data[label == lbl]
        for i, point in enumerate(class_data):
            save_as_ply(point, os.path.join(class_dir, f"point_{i}.ply"))

    if not num_point:
        return data[:, :, :], label
    else:
        return data[:, :num_point, :], label

"""
    # 选择少量的样本以减少训练时间
    if data.shape[0] > sample_size:
        indices = np.random.choice(data.shape[0], sample_size, replace=False)
        data = data[indices, :, :]
        label = label[indices, :]
        
    export_dir='C:/Users/29822/Desktop/try/model-40'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    for i, point_cloud in enumerate(data):
        ply_filename = os.path.join(export_dir, f"sample_{i}.ply")
        write_ply(point_cloud, ply_filename)
        
    
    for lbl in range(20):  # 前20个类别
        class_dir = os.path.join(save_dir, f"class_{lbl}")
        os.makedirs(class_dir, exist_ok=True)
"""