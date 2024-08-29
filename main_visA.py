import torch
from dataset.dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset.dataset import MVTecDataset,MVTecDataset_no_seg,visADataset,visADataset_no_seg
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation_visA,evaluation_me, evaluation_visualization, evaluation, evaluation_visualization_no_seg
from torch.nn import functional as F
import argparse
import sys

# Set random seed
def setup_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Loss function, can be used for ablation studies
def loss_function(a, b, L2):  # Input two tensor arrays
    cos_loss = torch.nn.CosineSimilarity()
    #print(a[0].size())  # a[0] = [16,256,64,64]
    #print(a[1].size())  # a[1] = [16,512,32,32]
    #print(a[2].size())  # a[2] = [16,1024,16,16]
    loss = 0

    # Use cosine loss
    if L2 == 0:
        for item in range(len(a)):  # For each tensor in a
            #print(torch.mean((1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))))
            loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))

    # Use L2 loss + cosine loss
    if L2 == 2:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += 0.5 * torch.mean(l2_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))       
             loss += 0.5 * torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))
    loss2 = loss_function_2(a,b)

    # Use L2 loss
    if L2 == 1:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += torch.mean(l2_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))       
    loss2 = loss_function_2(a,b)
    #print(loss)
    #print(loss2) 
    #sys.exit()
    return loss,loss2

# Try to calculate inter-group consistency loss
def loss_function_2(a, b):  # Input two tensor arrays
    mse_loss = torch.nn.MSELoss()
    # Compare results obtained by upsampling a2 and b2 with results obtained without upsampling a1 and b1
    a2 = F.interpolate(a[2], size=32, mode='bilinear', align_corners=True)
    b2 = F.interpolate(b[2], size=32, mode='bilinear', align_corners=True)
    l2 = torch.mean(mse_loss(a2.view(a2.shape[0],-1), b2.view(b2.shape[0],-1)))
    l1 = torch.mean(mse_loss(a[1].view(a[1].shape[0],-1), b[1].view(b[1].shape[0],-1)))
    loss2_1 = torch.abs(l2-l1)

    # Compare results obtained by upsampling a1 and b1 with results obtained without upsampling a0 and b0
    l0 = torch.mean(mse_loss(a[0].view(a[0].shape[0],-1), b[0].view(b[0].shape[0],-1)))

    a1 = F.interpolate(a[1], size=64, mode='bilinear', align_corners=True)
    b1 = F.interpolate(b[1], size=64, mode='bilinear', align_corners=True)
    l1 = torch.mean(mse_loss(a1.view(a1.shape[0],-1), b1.view(b1.shape[0],-1)))
    loss2_2 = torch.abs(l1-l0)

    #print(loss2_1,loss2_2)
    #sys.exit()
    loss2 = loss2_1 + loss2_2
    return loss2

def train(class_, epochs, learning_rate, res, batch_size, print_epoch, seg, data_path, save_path, print_canshu, score_num, print_loss, img_path, vis, cut, layerloss, rate, print_max, net, L2, seed): 
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(class_)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    data_transform, gt_transform = get_data_transforms(image_size, image_size) 

    train_path = data_path + class_ + '/train'
    test_path = data_path + class_ 
    ckp_path = save_path + net + class_ 

    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    # Whether to use segmentation
    if seg == 0:  
        test_data = visADataset_no_seg(root=test_path, transform=data_transform, phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    if seg == 1:
        test_data = visADataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    # Choose which network to use
    if net == 'wide_res50':   
        encoder,bn = wide_resnet50_2(pretrained=True)  # Encoder
        encoder = encoder.to(device)
        bn = bn.to(device)
        encoder.eval()  # Fix encoder model parameters    
        decoder = de_wide_resnet50_2(pretrained=False)  # Decoder, inverse structure of encoder
        decoder = decoder.to(device)
    if net == 'res18':
        encoder = resnet18(pretrained=True)  # Encoder
        encoder = encoder.to(device)
        encoder.eval()  # Fix encoder model parameters    
        decoder = de_resnet18(pretrained=False)  # Decoder, inverse structure of encoder
        decoder = decoder.to(device)
    if net == 'res34':
        encoder = resnet34(pretrained=True)  # Encoder
        encoder = encoder.to(device)
        encoder.eval()  # Fix encoder model parameters    
        decoder = de_resnet34(pretrained=False)  # Decoder, inverse structure of encoder
        decoder = decoder.to(device)
    if net == 'res50':
        encoder = resnet50(pretrained=True)  # Encoder
        encoder = encoder.to(device)
        encoder.eval()  # Fix encoder model parameters    
        decoder = de_resnet50(pretrained=False)  # Decoder, inverse structure of encoder
        decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))  # Pass a list of parameters to be optimized

    max_auc = []
    max_auc_epoch = []
    max_p_auc = []
    max_p_auc_epoch = []
    best_avg_score = 0

    # Start training
    for epoch in range(epochs):
        decoder.train()
        bn.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)  

            # Choose loss function  
            if layerloss == 0:
                loss = loss_function(inputs[0:3], outputs, L2)[0] 
            if layerloss == 1:
                loss = loss_function(inputs[0:3], outputs, L2)[0] + rate * loss_function(inputs[0:3], outputs, L2)[1]

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item()) 

        if print_loss == 1:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % print_epoch == 0:
            # Test set without mask
            if seg == 0:
                auroc_sp= evaluation_me(encoder, bn, decoder, res, test_dataloader, device, print_canshu, score_num)
                print('epoch:', (epoch + 1))
                print('Sample Auroc{:.3f}'.format(auroc_sp))
                max_auc.append(auroc_sp)
                max_auc_epoch.append(epoch + 1)
                if print_max == 1:
                    print('max_auc = ', max(max_auc))
                    print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('------------------')

                # Save model only if Sample AUROC is the maximum
                current_avg_score = auroc_sp

                if current_avg_score > best_avg_score:
                    print(f"New best model found at epoch {epoch+1} with Sample Auroc{auroc_sp:.3f}")
                    torch.save({'bn': bn.state_dict(),'decoder': decoder.state_dict()}, ckp_path + str(epoch+1) + str(seed) + 'sample_auc=' + str(auroc_sp) + '.pth')
                    best_avg_score = current_avg_score
               
                if vis == 1:  # Visualization output when no mask
                    evaluation_visualization_no_seg(encoder, decoder, res, test_dataloader, device, print_canshu, score_num, img_path)

            # Test set with mask and need localization
            if seg == 1:
                # Go through normal process
                # Plot
                if vis == 1:
                    evaluation_visualization(encoder, decoder, res, test_dataloader, device, print_canshu, score_num, img_path)
                # This part calculates the basic results and saves the results of the current epoch.
                auroc_px, auroc_sp = evaluation_visA(encoder,bn, decoder, res, test_dataloader, device, img_path)
                print('Pixel Auroc: {:.3f}, Sample Auroc: {:.3f}'.format(auroc_px, auroc_sp))


                # Update AUROC and AUPRO lists
                max_auc.append(auroc_sp)
                max_auc_epoch.append(epoch + 1)
                max_p_auc.append(auroc_px)
                max_p_auc_epoch.append(epoch + 1)


                # Print maximum AUROC and AUPRO, and the corresponding epoch
                print('max_auc_sample = ', max(max_auc))
                print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('max_auc_pixel = ', max(max_p_auc))
                print('max_epoch = ', max_p_auc_epoch[max_p_auc.index(max(max_p_auc))])

                # Calculate the average score of the current epoch
                current_avg_score = (auroc_sp + auroc_px) / 2

                # Save model only if the average of AUROC and AUPRO is the maximum
                if current_avg_score > best_avg_score:
                    print(f"New best model found at epoch {epoch+1} with average score: {current_avg_score:.3f} (Sample Auroc{auroc_sp:.3f}/Pixel Auroc{auroc_px:.3f})")
                    torch.save(decoder.state_dict(), ckp_path + str(epoch+1) + str(seed) + 'sample_auc=' + str(max(max_auc)) +'pixel_auc=' + str(max(max_p_auc)) +'.pth')
                    best_avg_score = current_avg_score
    return auroc_sp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)  # Training epochs
    parser.add_argument('--res', default=3, type=int)  # Select the number of connections, can choose 1, 2, 3, which actually represents 0, 1, 2 connections
    parser.add_argument('--learning_rate', default=0.005, type=float)  # Learning rate
    parser.add_argument('--batch_size', default=16, type=int)  # Batch size
    parser.add_argument('--seed', default=[111,250,444,999,114514], nargs='+', type=int)  # Random seed
    parser.add_argument('--class_', default='all', type=str)  # Select sub-dataset
    parser.add_argument('--seg', default=0, type=int)  # Choose whether segmentation is needed
    parser.add_argument('--print_epoch', default=50, type=int)  # Print every few epochs
    parser.add_argument('--data_path', default='/home/intern24/mvtec/', type=str)  # Path to dataset folder
    parser.add_argument('--save_path', default='/home/intern24/anomaly_checkpoints/dat_train2/skipconnection/', type=str)  # Path to save model files
    parser.add_argument('--print_canshu', default=1, type=int)  # Whether to print anomaly scores for test set
    parser.add_argument('--score_num', default=1, type=int)  # Number of anomaly scores used in the final anomaly score
    parser.add_argument('--print_loss', default=1, type=int)
    parser.add_argument('--img_path', default='/home/intern24/anomaly_checkpoints/dat_train2/skipconnection/result_img/', type=str)  # If segmentation is needed, select the path
    parser.add_argument('--vis', default=0, type=int)  # If segmentation is needed, whether to visualize output
    parser.add_argument('--cut', default=0, type=int)  # Whether to use cutpaste data augmentation
    parser.add_argument('--layerloss', default=1, type=int)  # Whether to use inter-group consistency loss
    parser.add_argument('--rate', default=0.05, type=float)  # Proportion of inter-group consistency loss
    parser.add_argument('--print_max', default=1, type=int)  # Whether to print the best AUC
    parser.add_argument('--net', default='wide_res50', type=str)  # Available net types, can choose res18, res34, res50, wide_res50
    parser.add_argument('--L2', default=0, type=int)  # Whether to use L2 loss function
    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    if args.class_ == 'all':
        all = [ 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
               'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
              'macaroni1','macaroni2']
        epoch_ = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        rate_ = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
        for class_, epoch, rate in zip(all, epoch_, rate_):
            print(class_)
            print(epoch)
            print(rate)
            print_epoch = args.print_epoch
            seed = args.seed[0]
            print('*************************')
            print('seed:', seed)
            setup_seed(seed)
            train(class_, epoch, args.learning_rate, args.res, args.batch_size, print_epoch, args.seg, args.data_path, args.save_path, args.print_canshu, args.score_num, args.print_loss, args.img_path, args.vis, args.cut, args.layerloss, rate, args.print_max, args.net, args.L2, seed)
            print('*************************')  

    if args.class_ != 'all':
            for seed in args.seed:
                print('*************************')
                print('seed:', seed)
                setup_seed(seed)
                train(args.class_, args.epochs, args.learning_rate, args.res, args.batch_size, args.print_epoch, args.seg, args.data_path, args.save_path, args.print_canshu, args.score_num, args.print_loss, args.img_path, args.vis, args.cut, args.layerloss, args.rate, args.print_max, args.net, args.L2, seed)
                print('*************************') 