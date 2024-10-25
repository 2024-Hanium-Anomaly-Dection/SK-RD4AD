# Skip-Connected Reverse Distillation for Robust One-Class Anomaly Detection

We introduce **SK-RD4AD**, an advanced anomaly detection model built upon the **Reverse Distillation for Anomaly Detection (RD4AD)** framework. SK-RD4AD enhances the RD4AD model by incorporating **non-corresponding skip connections** between layers, effectively mitigating deep feature loss while preserving multi-scale feature representations, thereby improving anomaly detection performance significantly.

## FrameWork

The original RD4AD model struggled with information loss in deep layers, particularly when handling multi-scale features. Our approach integrates **non-corresponding skip connections** between encoder and decoder layers, ensuring that feature maps from earlier stages of the encoder are preserved and passed directly to deeper decoder layers. This architecture allows the model to capture both fine details and high-level semantic information, which are critical for accurately detecting subtle anomalies.

![image](https://github.com/user-attachments/assets/64b2f6de-1ec1-4232-a86c-28a4f5836b3e)

### Key Improvements in Our Model:
1. **Non-corresponding Skip Connections**: We introduce connections between non-corresponding layers (e.g., E3 to D2, E1 to D2), preserving both low- and high-level features for more effective reconstruction.
2. **Multi-Scale Feature Preservation**: The network can now better handle subtle anomalies by retaining fine-grained and high-level details simultaneously.

### Advantages of Our Model:
- **Information Preservation**: Features from each encoder layer are directly passed to later decoder layers, minimizing information loss.
- **Enhanced Feature Utilization**: Multi-scale features from various levels of the encoder are efficiently reused in the decoder, improving reconstruction accuracy.

## Experiment Results

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the MVTec and VAD datasets from:
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- [Valeo Anomaly Dataset (VAD)](https://drive.google.com/file/d/1LbHHJHCdkvhzVqekAIRdWjBWaBHxPjuu/view/)

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

The MVTec dataset results demonstrate that the SK-RD4AD model significantly outperforms the baseline RD4AD model in anomaly detection. With a total average performance of **98.06 / 94.69**, SK-RD4AD showcases its **enhanced capability to detect subtle anomalies** across various categories. This improvement highlights its **effectiveness in capturing nuanced features**, making it a valuable solution for industrial applications requiring precise anomaly detection. Overall, SK-RD4AD's advancements emphasize its potential in addressing complex anomaly detection challenges.

### 5. Valeo Dataset Performance

| Setting   | Baseline AUROC | SK-RD4AD AUROC |
|-----------|----------------|----------------|
| One-Class | 84.5           | **87.0**       |

The SK-RD4AD model demonstrates **significant enhancements in performance on the Valeo Anomaly Dataset (VAD)**, achieving an AUROC of **87.0**, compared to the baseline AUROC of **84.5**. This improvement highlights the model's robustness and effectiveness in detecting fine-grained anomalies in **real-world applications**. The results further validate SK-RD4AD's capability to generalize across different datasets, reinforcing its potential as a powerful tool for anomaly detection tasks. Overall, the performance gains on the VAD dataset underscore the model's strengths in handling complex and subtle anomaly scenarios.


### 6. Visualization
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
