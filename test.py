import torch
from dataset.dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset.dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import os
from skimage.segmentation import mark_boundaries

plt.switch_backend('agg')

# Calculate anomaly score map
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])  # Initialize anomaly map
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):  # Iterate over anomaly maps
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)  # Compute cosine similarity between feature maps
        a_map = torch.unsqueeze(a_map, dim=1)  # Unsqueeze to add an additional dimension
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)  # Upsample to output size
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map  # Multiply anomaly maps
        else:
            anomaly_map += a_map  # Sum anomaly maps
    return anomaly_map, a_map_list  # Return final anomaly map and list of maps

# Visualize anomaly map over image
def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# Normalize the image between 0 and 1
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

# Convert image to heatmap
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

# Compute Per-Region Overlap (PRO) and Area Under the Curve (AUC)
def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlapping (PRO) and 0 to 0.3 FPR
    Args:
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"

    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df,pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

# Evaluation function without segmentation
def evaluation_me(encoder, bn, decoder, res, dataloader, device, print_canshu, score_num):
    decoder.eval() 
    bn.eval()
    encoder.eval()
    
    # Lists to store sample-level labels and predictions
    gt_list_sp = [] 
    pr_list_sp = [] 


    with torch.no_grad():
        for (img, label, _) in dataloader:
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)
            
            # Calculate final anomaly map
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # Apply Gaussian smoothing

            # Add sample-level labels
            gt_list_sp.append(label.numpy()[0]) 

            # Calculate sample-level predictions
            pre_map = np.flipud(np.sort(anomaly_map.flatten()))
            pre = 0
            for x in range(score_num):
                pre +=pre_map[x]
            pre = pre/score_num
            pr_list_sp.append(round(pre,3))

        if print_canshu == 1:
            print(gt_list_sp, pr_list_sp)  # Print intermediate results

        # Calculate sample-level AUROC
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        
    
    return auroc_sp

# Generate heatmaps for evaluation visualization
def evaluation_visualization(encoder, decoder, res, dataloader, device, print_canshu, score_num, img_path):
    count = 0
    decoder.eval()
    with torch.no_grad():
        for img, gt, label, _, ip in dataloader:
            print(ip[0][-20:-4])
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3], inputs[0:3], res)  

            anomaly_map, amap_list = cal_anomaly_map([inputs[0:3][-1]], [outputs[-1]], img.shape[-1], amap_mode='a')  # Generate anomaly map
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # Apply Gaussian filter
            ano_map = min_max_norm(anomaly_map)  # Normalize data

            ano_map = cvt2heatmap(255-ano_map*255)  # Convert to heatmap
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)

            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)  # Overlay heatmap on original image

            # Plot heatmap
            plt.subplot(1,3,1)
            plt.imshow(ano_map)
            plt.axis('off')

            # Plot ground truth
            gt = gt.cpu().numpy().astype(int)[0][0]*255
            plt.subplot(1,3,2)
            plt.imshow(gt, cmap='gray')
            plt.axis('off')

            # Plot original image
            plt.subplot(1,3,3)
            plt.imshow(img)
            plt.axis('off')

            if (os.path.exists(img_path) == 0):
                os.mkdir(img_path)

            # Save image
            plt.savefig(img_path + str(ip[0][-20:-4]).replace('/', '_') + '.png')

            count += 1

# Generate heatmaps for evaluation visualization without segmentation
def evaluation_visualization_no_seg(encoder, decoder, res, dataloader, device, print_canshu, score_num, img_path):
    count = 0
    decoder.eval()
    with torch.no_grad():
        for img, label, _  in dataloader:
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3], inputs[0:3], res)  

            anomaly_map, amap_list = cal_anomaly_map([inputs[0:3][-1]], [outputs[-1]], img.shape[-1], amap_mode='a')  # Generate anomaly map
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # Apply Gaussian filter
            ano_map = min_max_norm(anomaly_map)  # Normalize data

            ano_map = cvt2heatmap(255-ano_map*255)  # Convert to heatmap
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)

            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)  # Overlay heatmap on original image

            # Plot heatmap
            plt.subplot(1,2,1)
            plt.imshow(ano_map)
            plt.axis('off')

            # Plot original image
            plt.subplot(1,2,2)
            plt.imshow(img)
            plt.axis('off')

            if (os.path.exists(img_path) == 0):
                os.mkdir(img_path)

            # Save image
            plt.savefig(img_path + str(count).replace('/', '_') + '.png')

            count += 1

# Evaluation with segmentation, very time-consuming
def evaluation(encoder,bn, decoder, res, dataloader, device, img_path):
    decoder.eval()
    bn.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, _, _ in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res) 
            # Compute anomaly maps using encoder's first three outputs and decoder's outputs
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # Apply Gaussian filter

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.int()

            #unique_values = torch.unique(gt)
            #print("Unique values in gt:", unique_values)

            if label.item() != 0:
                # print(gt.squeeze(0).cpu().numpy().astype(int))
                # print(set(gt.flatten()))
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))

            # Convert multi-dimensional arrays to one-dimensional arrays
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)


# Evaluation with segmentation, very time-consuming
def evaluation_visA(encoder, bn, decoder, res, dataloader, device, img_path):
    decoder.eval()
    bn.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    # aupro_list = []
    with torch.no_grad():
        for img, gt, label, _, _ in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res) 
            # Compute anomaly maps using encoder's first three outputs and decoder's outputs
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # Apply Gaussian filter

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.int()

            #unique_values = torch.unique(gt)
            #print("Unique values in gt:", unique_values)

            # if label.item() != 0:
            #     # print(gt.squeeze(0).cpu().numpy().astype(int))
            #     # print(set(gt.flatten()))
            #     aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
            #                                   anomaly_map[np.newaxis, :, :]))

            # Convert multi-dimensional arrays to one-dimensional arrays
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp#, round(np.mean(aupro_list), 3)