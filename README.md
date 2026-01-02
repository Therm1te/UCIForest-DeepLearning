```markdown
# 🌲 UCI Forest CoverType: Deep Learning vs. Ensemble Methods

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

A high-performance Deep Learning project classifying forest cover types based on cartographic variables (elevation, soil type, etc.). This project implements a **Custom MLP Architecture** achieving ~93-94% accuracy, comparable to industry-standard Ensemble methods, through rigorous feature engineering and architecture tuning.

---

## 📊 Key Results

| Model | Accuracy | Key Technique |
| :--- | :--- | :--- |
| **Custom MLP (Neural Net)** | **93.45%** | Selective Scaling + Wide Architecture |
| **Random Forest** | **95.12%** | Orthogonal Decision Boundaries |
| *Standard MLP Baseline* | *~91.00%* | *Standard Scaling (Naive)* |

> **Note:** While the Random Forest performed slightly better "out of the box," the optimized MLP pushed the boundaries of what Deep Learning can achieve on tabular data by solving specific data distribution challenges.

---

## 🧠 The Challenge

The **UCI Forest CoverType dataset** contains ~581,000 instances with 54 columns predicting 7 different forest cover types. The challenge lies in:
1.  **Severe Class Imbalance:** Classes 1 & 2 dominate; Class 4 is extremely rare.
2.  **Mixed Data Types:** A combination of continuous vars (Elevation, Slope) and binary vars (Soil Types, Wilderness Areas).
3.  **"Lodgepole" Conflict:** High feature overlap between Spruce/Fir (Class 1) and Lodgepole Pine (Class 2).

---

## ⚙️ The "Golden" Workflow (Methodology)

To achieve >93% accuracy with a Neural Network, standard preprocessing was insufficient. This project utilized a **Selective Preprocessing Pipeline**:

### 1. Robust Data Loading
- Switched to `fetch_openml` to bypass UCI server HTTP 403 errors.
- Enforced strict type casting (`float32` for features, `int` for targets) to prevent memory overhead and TensorFlow compatibility issues.

### 2. Selective Scaling (The Difference Maker)
Most pipelines blindly scale all features. This project applied a hybrid approach:
- **Columns 0–9 (Continuous):** Applied `StandardScaler`.
- **Columns 10–54 (Binary):** **Left untouched.** Scaling binary data destroys its semantic meaning (0 vs 1) and introduces noise.
- *Result:* This specific step improved accuracy from ~91% to ~94%.

### 3. Stratified Splitting
- Used `stratify=y` during Train/Val/Test splitting to ensure rare classes (e.g., Cottonwood/Willow) were represented in the validation set.

---

## 🏗️ Model Architecture

The final model is a **Wide, Regularized MLP** designed for tabular feature interaction:

- **Input:** 54 Features
- **Hidden Layer 1:** 256 Neurons | L2 Reg | Batch Norm | ReLU | Dropout (0.1)
- **Hidden Layer 2:** 128 Neurons | L2 Reg | Batch Norm | ReLU | Dropout (0.1)
- **Output:** Softmax (7 Classes)
- **Optimizer:** Adam with `ReduceLROnPlateau` and `EarlyStopping`.

---

## 📉 Comparative Analysis: MLP vs. Random Forest

### Why Random Forest Won (Slightly)
The dataset consists of distinct thresholds (e.g., *Elevation > 3000m*). Tree-based models naturally learn these **orthogonal decision boundaries**. Neural Networks, which approximate functions smoothly, struggle to create the sharp "cutoffs" required to separate the dominant classes (Spruce vs. Pine) without massive feature engineering.

### Confusion Matrix Insight
The primary error source for the Neural Network was distinguishing **Class 1 (Spruce/Fir)** from **Class 2 (Lodgepole Pine)**. These species occupy similar ecological niches, leading to overlapping feature distributions that the MLP found difficult to disentangle compared to the Random Forest.

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/uci-forest-cover-dl.git
   cd uci-forest-cover-dl
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Notebook**
   ```bash
   jupyter notebook DL_UCIForestCoverType_AbdullahAsim.ipynb
   ```

---

## 📂 Project Structure

```
├── DL_UCIForestCoverType_AbdullahAsim.ipynb    # Main pipeline (Loading, Preprocessing, Training, Eval)
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
```

---

## 🔮 Future Improvements

Based on the error analysis, future iterations could include:
1.  **Feature Engineering:** Explicitly creating interaction terms (e.g., `Elevation * Hillshade`) to help the MLP separate Class 1 and 2.
2.  **TabNet Implementation:** Utilizing attention-based architectures designed specifically for tabular data.
3.  **Global Seeding:** Implementing `os.environ['PYTHONHASHSEED']` and `tf.random.set_seed` for bit-exact reproducibility.
```