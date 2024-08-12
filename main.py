# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset.dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset.dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F
import logging

from model.dat import DeformableAttention2D



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_function_cross(a, b):
    # mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if item == 2:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[3].view(b[3].shape[0], -1)))
        elif item == 3:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[2].view(b[2].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(_class_):
    # 로깅 설정
    logging.basicConfig(filename=f'/home/intern24/anomaly/input_dat_encoder2/AnomalyDetection/output_log/training_dat_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f'Training started for class: {_class_}')

    epochs = 200
    dat_lr = 0.001
    learning_rate = 0.005
    batch_size =16
    image_size = 256
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = '/home/intern24/mvtec/' + _class_ + '/train'
    test_path = '/home/intern24/mvtec/' + _class_  
    ckp_path = '/home/intern24/anomaly_checkpoints/dat_train/' + 'input_dat_add_'+_class_+'.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    # dat 학습 가능하도록 설정
    # for param in encoder.dat.parameters():
    #     param.requires_grad = True

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    dat = DeformableAttention2D().to(device)

    #dat 모듈 최적화 진행
    optimizer_dat = torch.optim.Adam(list(dat.parameters()), lr=dat_lr, betas=(0.5,0.999))
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))


    for epoch in range(epochs):
        
        dat.train()
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)

            input_dat = dat(inputs[2]) 
            inputs = [inputs[0], inputs[1], inputs[2], input_dat]

            outputs = decoder(bn(inputs))
            loss = loss_function_cross(inputs, outputs)
            optimizer_dat.zero_grad()
            optimizer.zero_grad()
            loss.backward()

            optimizer_dat.step()
            optimizer.step()

            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {np.mean(loss_list):.4f}')
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, dat, bn, decoder, test_dataloader, device)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3f}'.format(auroc_px, auroc_sp, aupro_px))
            logging.info(f'Pixel Auroc: {auroc_px:.3f}, Sample Auroc: {auroc_sp:.3f}, Pixel Aupro: {aupro_px:.3f}')
            
            torch.save({'encoder' : dat.state_dict(),
                        'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px




if __name__ == '__main__':

    setup_seed(111)
    item_list = ['bottle','carpet' , 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        train(i)

