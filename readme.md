## Anomaly Detection via Reverse Distillation
we develop Anomaly Detection via Reverse Distilation from [paper](https://arxiv.org/abs/2201.10703/)

1. Environment
```Shell
    pip install -r requirements.txt
```
2. Dataset
    > You should download MVTec from [MVTec AD: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/). The folder "mvtec" should be unpacked into the code folder.
3. Train and Test the Model
We have write both training and evaluation function in the main.py, execute the following command to see the training and evaluation results.
```Shell
    python main.py 
    \--epochs 200 \--res 3 \--learning_rate 0.005 \--batch_size 16 \--seed 111 \--class_ all \--seg 1 \--print_epoch 10 \--data_path /home/intern24/mvtec/ \--save_path /home/intern24/anomaly_checkpoints/dat_train2/skipconnection/ \--print_canshu 1 \--score_num 1 \--print_loss 1 \--img_path /home/intern24/anomaly_checkpoints/dat_train2/skipconnection/result_img/ \--vis 0 \--cut 0 \--layerloss 1 \--rate 0.1 \--print_max 1 \--net wide_res50 \--L2 0
```
    
 ## Reference
	@InProceedings{Deng_2022_CVPR,
    author    = {Deng, Hanqiu and Li, Xingyu},
    title     = {Anomaly Detection via Reverse Distillation From One-Class Embedding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9737-9746}}
