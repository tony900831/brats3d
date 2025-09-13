Here is the concise, resume-ready experiment report in both English and Chinese.

---

### **Brain Tumor Segmentation Experiment Report**

#### **English Version**

##### 1. Result

Achieved a best validation overall Dice score of **0.7466** for BraTS2020 brain tumor segmentation. Per-class Dice scores were: Necrotic/Non-enhancing Tumor Core (NCR): 0.7139, Peritumoral Edema (ED): 0.8071, Enhancing Tumor (ET): 0.7188.

##### 2. Setup

A 3D ResUNet+SE (base=32) model was trained on 295 cases and validated on 73 cases (fixed split). Training utilized Automatic Mixed Precision (AMP=bf16) with a combined loss of CrossEntropy (label smoothing 0.05) and SoftDice (excluding background). Validation employed a sliding-window approach (patch=128^3, overlap=0.5) without Test-Time Augmentation (TTA). Early stopping and ReduceLROnPlateau were based on validation overall Dice.

##### 3. Observations

The model trained for 66 epochs, reaching its best performance at epoch 46. Total training time was approximately 35.7 hours. Peritumoral Edema (ED) consistently showed the highest Dice score (0.8071), indicating robust segmentation for this subregion compared to others.

##### 4. Reproducibility

The experiment utilized the BraTS2020 dataset with a fixed 295/73 train/validation split (`split.json`). Model architecture (3D ResUNet+SE), key training parameters (AMP=bf16, specific loss functions), and validation strategy (sliding-window, no TTA) are clearly defined to ensure full reproducibility.

---

#### **中文版本**

##### 1. 結果

在BraTS2020腦腫瘤分割任務中，驗證集最佳整體Dice係數達到**0.7466**。各類別Dice係數為：壞死/非強化腫瘤核心 (NCR): 0.7139, 腫瘤周圍水腫 (ED): 0.8071, 強化腫瘤 (ET): 0.7188。

##### 2. 設定

採用3D ResUNet+SE模型 (base=32)，在295個訓練病例和73個驗證病例上進行訓練 (固定劃分)。訓練使用自動混合精度 (AMP=bf16)，損失函數結合交叉熵 (標籤平滑0.05) 和SoftDice (排除背景)。驗證採用滑動窗口方法 (patch=128^3, 重疊0.5)，無測試時增強 (TTA)。早停和學習率調整 (ReduceLROnPlateau) 基於驗證集整體Dice係數。

##### 3. 觀察

模型共訓練66個epoch，在第46個epoch達到最佳性能。總訓練時長約35.7小時。腫瘤周圍水腫 (ED) 類別的Dice係數最高 (0.8071)，表明該區域的分割效果較為穩健。

##### 4. 可重現性

實驗使用BraTS2020數據集及固定的295/73訓練/驗證劃分 (`split.json`)。模型架構 (3D ResUNet+SE)、關鍵訓練參數 (AMP=bf16, 指定損失函數) 和驗證策略 (滑動窗口, 無TTA) 已詳細說明，確保實驗可完全重現。