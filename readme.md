# SK-RD4AD : RD4AD Enhanced Model with Skip Connections
we develop Anomaly Detection via Reverse Distilation from [paper](https://arxiv.org/abs/2201.10703/)


## Model Architecture Overview

### Baseline Limitations and Our Enhancements
The original RD4AD model, as noted by its creators, faced limitations in effectively utilizing multi-scale feature maps and suffered from information loss at deeper network layers. These issues limited its performance, especially in complex and varied datasets.

To overcome these challenges, we introduced **skip connections** in our model. This enhancement allows features extracted at each encoder layer to be directly transmitted to corresponding layers in the decoder, minimizing information loss and maximizing feature reuse.

![image](https://github.com/user-attachments/assets/c702dfc2-d5a7-42bb-8f6a-12812c533f66)


The diagram above illustrates our modified RD4AD model, SK-RD4AD , which integrates skip connections at multiple layers. This architecture ensures that features extracted at each encoder layer are not only passed down through the network but are also directly utilized in the corresponding decoder layers. This helps in mitigating information dilution and loss commonly seen in deep networks.
Our model incorporates skip connections to overcome the limitations identified by the original RD4AD authors, where the network's inability to effectively utilize varying feature map sizes and information loss in deeper layers was causing performance bottlenecks.

#### Advantages of Our Approach
- **Information Preservation**: Outputs from each layer are not only propagated forwards but are also sent directly to the matching decoder layers. This approach significantly reduces the dilution and loss of information commonly seen in deep networks.
- **Feature Reutilization**: Features extracted at various levels in the encoder are directly reused in the decoder, allowing for more precise and detailed reconstruction of information. This is particularly effective in imaging contexts where fine details, such as textures, are crucial.

The integration of skip connections effectively addresses the core limitations of the original RD4AD model, allowing for more accurate detection of anomalies by preserving and utilizing detailed feature information throughout the network.

## Experiment Results
<details>
  <summary>Training</summary>
  1. Environment
```Shell
    pip install -r requirements.txt
```
2. Dataset
    > You should download MVTec , VAD from [MVTec AD: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/) , [VAD: Valeo Anomaly Dataset](https://drive.google.com/file/d/1LbHHJHCdkvhzVqekAIRdWjBWaBHxPjuu/view/).
3. Train and Test the Model
We have write both training and evaluation function in the main.py, execute the following command to see the training and evaluation results.
```Shell
    python main.py 
    \--epochs 200 \--res 3 \--learning_rate 0.005 \--batch_size 16 \--seed 111 \--class_ all \--seg 1 \--print_epoch 10 \--data_path /home/intern24/mvtec/ \--save_path /home/intern24/anomaly_checkpoints/dat_train2/skipconnection/ \--print_canshu 1 \--score_num 1 \--print_loss 1 \--img_path /home/intern24/anomaly_checkpoints/dat_train2/skipconnection/result_img/ \--vis 0 \--cut 0 \--layerloss 1 \--rate 0.1 \--print_max 1 \--net wide_res50 \--L2 0
```
</details>

### MVTec Dataset Performance

The table below summarizes the performance improvements on the MVTec dataset, demonstrating significant enhancements across various categories compared to the baseline, particularly in accurately capturing differences between normal and reconstructed images which leads to superior anomaly detection performance.

| Category     | Baseline (Pixel AUROC/AUPRO) | Ours (Pixel AUROC/AUPRO)   |
|--------------|------------------------------|----------------------------|
| Carpet       | 98.9 / 97.0                  | **99.0 / 97.0**            |
| Bottle       | 98.7 / 96.6                  | **98.9 / 96.8**            |
| Hazelnut     | 98.9 / 95.5                  | **99.0 / 95.6**            |
| Leather      | 99.4 / 99.1                  | **99.4 / 99.0**            |
| Cable        | 97.4 / 91.0                  | **97.4 / 90.7**            |
| ...          | ...                          | ...                        |
| **Total Avg**| 97.8 / 93.9                  | **97.91 / 94.33**          |

### VAD Dataset Performance

On challenging datasets like Valeo, the introduction of skip connections further proved its effectiveness, enabling the network to capture more detailed and nuanced information which is critical for detecting subtle anomalies.

| Setting      | Baseline (Sample AUROC) | Ours (Sample AUROC)        |
|--------------|-------------------------|----------------------------|
| One-Class    | 84.5                    | **87.0**                   |

The results on complex datasets such as Valeo underline the enhanced adaptability and precision of our model due to the structural improvements, confirming that our approach not only retains but also significantly leverages detailed features for anomaly detection.


## Conclusion
By introducing skip connections, our RD4AD model not only addresses the previously identified limitations but also sets a new standard in anomaly detection capabilities, particularly in industrial applications where accuracy and detail are paramount. This model is particularly potent in environments where precise anomaly localization and characterization are crucial for quality control and maintenance.

    
 ## Reference
	@InProceedings{Deng_2022_CVPR,
    author    = {Deng, Hanqiu and Li, Xingyu},
    title     = {Anomaly Detection via Reverse Distillation From One-Class Embedding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9737-9746}}

	@misc{baitieva2024supervised,
      title={Supervised Anomaly Detection for Complex Industrial Images}, 
      author={Aimira Baitieva and David Hurych and Victor Besnier and Olivier Bernard},
      year={2024},
      eprint={2405.04953},
      archivePrefix={arXiv},
      primaryClass={cs.CV}}




