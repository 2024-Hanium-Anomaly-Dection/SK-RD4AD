# Skip-Connected Reverse Distillation for Robust One-Class Anomaly Detection

We introduce **SK-RD4AD**, an advanced anomaly detection model built upon the **Reverse Distillation for Anomaly Detection (RD4AD)** framework. SK-RD4AD enhances the RD4AD model by incorporating **non-corresponding skip connections** between layers, effectively mitigating deep feature loss while preserving multi-scale feature representations, thereby improving anomaly detection performance significantly.

## Framework

The **SK-RD4AD** model enhances the traditional RD4AD framework by addressing the critical challenge of **information loss** that occurs in deep layers, particularly when processing multi-scale features essential for accurate anomaly detection. The original RD4AD model struggled to retain and effectively utilize these features, leading to reduced performance in identifying subtle anomalies. To mitigate this, our model incorporates **non-corresponding skip connections** between encoder and decoder layers, allowing feature maps from early encoder stages to bypass intermediate layers and directly connect with deeper decoder layers.

This architecture design allows the model to capture both **fine-grained details** and **high-level semantic information**, which are crucial for distinguishing normal patterns from anomalous ones. By leveraging non-corresponding skip connections, SK-RD4AD not only preserves vital information across the network but also strengthens the feature flow, enabling better reconstruction of the input image and highlighting subtle anomalies.

![Architecture Diagram](https://github.com/user-attachments/assets/64b2f6de-1ec1-4232-a86c-28a4f5836b3e)

### Key Improvements in Our Model:
1. **Non-Corresponding Skip Connections**: We introduce selective connections between non-corresponding encoder and decoder layers (e.g., E3 to D2, E1 to D2). These links allow high-resolution, early-stage features to bypass intermediate processing and directly contribute to deeper decoding stages. This process preserves both low-level and high-level features, ensuring that fine details remain intact alongside broader semantic structures. This approach enhances the model’s sensitivity to subtle deviations within complex scenes.
   
2. **Multi-Scale Feature Preservation**: With the integration of multi-scale skip connections, SK-RD4AD effectively retains and combines both fine-grained and high-level details, which are processed in parallel across various network depths. This multi-scale feature preservation enables the model to capture intricate, localized anomalies as well as larger, structural irregularities, significantly improving its robustness in detecting subtle and diverse anomaly patterns.

### Advantages of Our Model:
- **Information Preservation**: By allowing features from each encoder layer to directly reach later decoder layers, SK-RD4AD minimizes information loss that typically occurs in deep-layered networks. This direct pathway for feature preservation ensures that critical details are not diluted through repeated transformations, resulting in a more accurate reconstruction and reliable anomaly identification.
  
- **Enhanced Feature Utilization**: The model reuses multi-scale features from various encoder stages in the decoder, maximizing the utility of each feature level. This targeted reuse of features ensures that the decoder receives a comprehensive, multi-layered perspective of the input data, leading to more precise anomaly localization and reconstruction accuracy.

By addressing the limitations of the original RD4AD model, SK-RD4AD establishes a robust architecture that is well-suited for high-performance anomaly detection in complex and industrial settings.

## Experiment Results

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the MVTec and VAD datasets from:
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- [Valeo Anomaly Dataset (VAD)](https://drive.google.com/file/d/1LbHHJHCdkvhzVqekAIRdWjBWaBHxPjuu/view/)
- [visA Dataset](https://github.com/amazon-science/spot-diff)

### 3. Train and Test the Model
To train and evaluate the model, run the following command:
```bash
python main.py \
    --epochs 200 \
    --res 3 \
    --learning_rate 0.005 \
    --batch_size 16 \
    --seed 111 \
    --class all \
    --seg 1 \
    --print_epoch 10 \
    --data_path /home/user/mvtec/ \
    --save_path /home/user/anomaly_checkpoints/skipconnection/ \
    --print_loss 1 \
    --net wide_res50 \
    --L2 0
```

### 4. MVTec Dataset Performance

| Category     | Baseline (Pixel AUROC/AUPRO) | SK-RD4AD (Pixel AUROC/AUPRO)   |
|--------------|------------------------------|--------------------------------|
| Carpet       | 98.9 / 97.0                  | **99.2 / 97.7**                |
| Bottle       | 98.7 / 96.6                  | **98.8 / 96.9**                |
| Hazelnut     | 98.9 / 95.5                  | **99.1 / 96.2**                |
| ....       | ... / ...                  | ... / ...               |
| Leather      | 99.4 / 99.1                  | **99.6 / 99.2**                |
| Total Avg    | 97.8 / 93.9                  | **98.06 / 94.69**              |

The MVTec dataset results demonstrate that the SK-RD4AD model significantly outperforms the baseline RD4AD model in anomaly detection. With a total average performance of **98.06 / 94.69**, SK-RD4AD showcases its **enhanced capability to detect subtle anomalies** across various categories. This improvement highlights its **effectiveness in capturing nuanced features**, making it a valuable solution for industrial applications requiring precise anomaly detection. SK-RD4AD’s ability to retain both fine-grained and high-level features is evident from the performance gains across a diverse set of categories, reinforcing its suitability for complex, real-world industrial tasks.

### 5. Valeo Anomaly Dataset Performance

| Setting   | Baseline AUROC | SK-RD4AD AUROC |
|-----------|----------------|----------------|
| One-Class | 84.5           | **87.0**       |

The SK-RD4AD model demonstrates **significant enhancements in performance on the Valeo Anomaly Dataset (VAD)**, achieving an AUROC of **87.0**, compared to the baseline AUROC of **84.5**. This improvement highlights the model's robustness and effectiveness in detecting fine-grained anomalies in **real-world applications**. The results further validate SK-RD4AD's capability to generalize across different datasets, reinforcing its potential as a powerful tool for anomaly detection tasks. The performance gains on the VAD dataset underscore the model's strengths in handling complex and subtle anomaly scenarios, making it an adaptable solution for diverse industrial applications.

### 6. VisA Dataset Performance

| Category     | RD4AD (Pixel AUROC / AUPRO) | SK-RD4AD (Pixel AUROC / AUPRO) |
|--------------|-----------------------------|--------------------------------|
| PCB1         | 99.6 / 43.2                 | 99.6 / **93.7**                |
| PCB2         | 98.3 / 46.4                 | 98.3 / **89.2**                |
| PCB3         | 99.3 / 80.3                 | 98.3 / **90.3**                |
| PCB4         | 98.2 / 72.2                 | **98.6 / 89.0**                |
| Pipe Fryum   | 99.1 / 68.3                 | 99.1 / **94.8**                |
| Candle       | 98.9 / 92.2                 | 98.6 / **93.9**                |
| Capsules     | 99.4 / 56.9                 | 99.1 / **91.9**                |
| Cashew       | 94.4 / 79.0                 | **98.1 / 87.3**                |
| Chewing Gum  | 97.6 / 92.5                 | **97.7 / 94.3**                |
| Fryum        | 96.4 / 81.0                 | **96.7 / 90.3**                |
| Macaroni1    | 99.3 / 71.9                 | 99.3 / **95.5**                |
| Macaroni2    | 99.1 / 68.0                 | **99.3 / 95.2**                |
| **Total Avg** | 97.8 / 70.9                 | **98.5 / 92.1**                |

The results on the VisA dataset show that SK-RD4AD achieves substantial performance gains over the baseline RD4AD, with an average AUROC/AUPRO improvement from **97.8 / 70.9** to **98.5 / 92.1**. These gains, particularly in categories such as PCB, Pipe Fryum, and Macaroni, highlight SK-RD4AD's capability to detect diverse and intricate anomalies with high accuracy. The significant improvements in AUPRO across categories confirm the model's ability to reliably capture subtle, complex anomalies, making it an ideal tool for industrial inspection tasks where precision is essential.


### 7. Visualization
<p align="center">
  <img src="https://github.com/user-attachments/assets/b2fe4e4b-6a4c-4c86-8caa-ebef8da92dd8" alt="figg3" width="45%">
  <img src="https://github.com/user-attachments/assets/dbbd9d8a-f70a-4a8f-9a9b-49e2f95ed4be" alt="ffig2" width="45%">

</p>

The visualization results demonstrate **the effectiveness of the SK-RD4AD model in detecting anomalies**. The anomaly maps highlight areas where the model identifies potential defects, using red and yellow hues to indicate regions of high confidence. The overlaid images combine the original images with the anomaly maps, clearly showing the detected anomalies' locations.


## Conclusion

SK-RD4AD effectively addresses deep feature loss by utilizing **non-corresponding skip connections**, enabling the model to better retain multi-scale features, particularly in challenging anomaly detection scenarios. This architecture enhances the model's ability to detect subtle anomalies, making SK-RD4AD a robust tool for anomaly detection across various industrial applications. Furthermore, the model demonstrates **state-of-the-art performance in one-class anomaly detection tasks**, achieving notable improvements in detection metrics compared to baseline models, thereby reinforcing its effectiveness in real-world scenarios. The results suggest that SK-RD4AD can serve as a **powerful solution** for applications requiring precise anomaly detection and robust feature preservation.

## References

```
@InProceedings{Deng_2022_CVPR,
    author    = {Deng, Hanqiu and Li, Xingyu},
    title     = {Anomaly Detection via Reverse Distillation From One-Class Embedding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
    pages     = {9737-9746}
}

@misc{baitieva2024supervised,
    title={Supervised Anomaly Detection for Complex Industrial Images}, 
    author={Aimira Baitieva and David Hurych and Victor Besnier and Olivier Bernard},
    year={2024},
    eprint={2405.04953},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
