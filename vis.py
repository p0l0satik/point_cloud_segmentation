import open3d as o3d
import numpy as np
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from src.loader.normals_loader import *
from normals_unet_v1_stock import *

def test_loader(loader, iter = 1):
    print(len(loader))
    i = 0
    device = "cuda:0"
    model, _, _ = get_model_and_optimizer(device=device)
    model.load_state_dict(torch.load("chpt/model_simple_normals_long_2_cross_up_right_unet_20230406_172753_19"))
    model.train(False)
    model.to(device)
    for data in loader:
        vinputs, vlabels, pcl, pcl_labels, idx = data
        idx = idx.squeeze()
        
        print(vinputs.shape, vlabels.shape, pcl.shape, pcl_labels.shape, idx.shape)
        # plt.figure(figsize=(20,80))
        # plt.imshow(vlabels[0], cmap="gray")

        # plt.figure(figsize=(20,80))
        # plt.imshow(vinputs[0][2], cmap="gray")
        pcd = o3d.geometry.PointCloud()
        points = pcl.detach().numpy().squeeze()
        print(points.shape)

        colors =  np.zeros_like(points)
        # self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        print(np.asarray(np.unique(pcl_labels,return_counts=True)).T)
        pcl_labels[pcl_labels > 0] = 1
        # pcl_labels[pcl_labels <= 0] = 0

        colors[:,0] = 255*pcl_labels.detach().numpy().squeeze()
        print(np.asarray(np.unique(pcl_labels,return_counts=True)).T)

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(np.asarray(np.unique(colors,return_counts=True)).T)
        o3d.visualization.draw_geometries([pcd])


        # o3d.visualization.draw_geometries([pcd])
        vinputs = torch.tensor(vinputs).to(device=device, dtype=torch.float)
        voutputs = model(vinputs)

        pcd2 = o3d.geometry.PointCloud()
        mask = idx >= 0
        print(idx, idx[:], idx[mask])
        points_back_proj = points[idx[mask]]
        colors_back_proj =  np.zeros_like(points_back_proj)
        print("back proj", points_back_proj.shape, colors_back_proj[:, 0].shape)
        vlabels = vlabels.squeeze()
        # print( np.asarray(np.unique(vlabels,return_counts=True)).T)
        colors_back_proj[:, 1] = 255*vlabels[mask].detach().numpy().squeeze()
        # print(np.asarray(np.unique(colors_back_proj[:,0],return_counts=True)).T)

        voutputs = F.softmax(voutputs, dim=1)[0].cpu().permute(1,2,0).detach().numpy().squeeze()[:,:, 1]
        # print( np.asarray(np.unique(vlabels,return_counts=True)).T)
        colors_back_proj[:, 2] = 255*voutputs[mask]

        pcd2.points = o3d.utility.Vector3dVector(points_back_proj)
        pcd2.colors = o3d.utility.Vector3dVector(colors_back_proj)
        o3d.visualization.draw_geometries([pcd2])

        plt.figure(figsize=(20,80))
        plt.imshow(vlabels)

        plt.figure(figsize=(20,80))
        plt.imshow(voutputs)
        plt.show()

        break



test_data = CustomKittiProcessingVis("/home/polosatik/mnt/kitty/dataset/sequences/00/", mode="test", proc_func=prep_long_2_right_up) 

test_loader_basic = DataLoader(test_data, batch_size=1, shuffle=True)

test_loader(test_loader_basic)