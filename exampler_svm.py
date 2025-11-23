# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 02:58:22 2025

@author: benker
"""

# -*- coding: utf-8 -*-
"""
Exemplar-SVM 物體檢測系統 (嚴格按照論文實現)
"""
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# ========== 1. 特徵提取 ==========
def extract_hog_features(image_path, resize_shape=(128,128)):
#提取HOG特徵 (與論文相同的8x8 cells, 2x2 blocks, 9 orientations)
    try:
        image = rgb2gray(imread(image_path))
        image_resized = resize(image, resize_shape, anti_aliasing=True)
        features, _ = hog(
            image_resized,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys',
            visualize=True
        )
        return features
    except Exception as e:
        print(f"特徵提取失敗：{image_path}")
        return None
# ========== 2. Exemplar-SVM 訓練 ==========
class ExemplarSVM:
    def __init__(self, C1=0.5, C2=0.01, n_iter=3, epochs=100, lr=0.01):
        """
        參數:
            C1: 正樣本懲罰係數 (0.5)
            C2: 負樣本懲罰係數 (0.01)
            n_iter: 硬負樣本挖掘迭代次數 (3次)
            epochs: 每次迭代的訓練輪數
            lr: 學習率
        """
        self.C1 = C1
        self.C2 = C2
        self.n_iter = n_iter
        self.epochs = epochs
        self.lr = lr
        self.w = None
        self.b = None
        self.calibrator = None

    def fit(self, x_E, X_neg):
        """訓練流程 (包含硬負樣本挖掘)"""
        # 初始負樣本集
        X_neg_current = X_neg.copy()
        
        for _ in range(self.n_iter):
            # 合併正負樣本 (1正+N負)
            X_train = np.vstack([x_E, X_neg_current])
            y_train = np.array([1] + [-1] * len(X_neg_current))
            
            # 訓練SVM
            self._train_svm(x_E, X_train, y_train)
            
            # 硬負樣本挖掘 (論文第3節)
            scores = X_neg @ self.w + self.b
            hard_neg_mask = (scores > -1.0)  # 找出被錯誤分類的負樣本
            X_neg_current = X_neg[hard_neg_mask]
            
            if len(X_neg_current) == 0:
                break

    def _train_svm(self, x_E, X, y):
        """SVM訓練 """
        n_features = X.shape[1]
        w = torch.zeros(n_features, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        optimizer = optim.Adam([w, b], lr=self.lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        x_E_tensor = torch.tensor(x_E, dtype=torch.float32)
        
        for _ in range(self.epochs):
            optimizer.zero_grad()
            
            # 正樣本損失 (論文式1第二項)
            pos_loss = self.C1 * torch.clamp(1 - (torch.dot(w, x_E_tensor) + b), min=0)
            
            # 負樣本損失 (論文式1第三項)
            neg_scores = -torch.matmul(X_tensor[y_tensor == -1], w) - b
            neg_loss = self.C2 * torch.sum(torch.clamp(1 - neg_scores, min=0))
            
            # L2正則化 (論文式1第一項)
            l2_penalty = 0.5 * torch.sum(w ** 2)
            
            total_loss = l2_penalty + pos_loss + neg_loss
            total_loss.backward()
            optimizer.step()
        
        self.w = w.detach().numpy()
        self.b = b.item()

    def calibrate(self, X_val, y_val):
        """校準步驟 (論文3.1節)"""
        scores = X_val @ self.w + self.b
        self.calibrator = LogisticRegression()
        self.calibrator.fit(scores.reshape(-1, 1), y_val)

    def predict(self, x):
        """預測 (包含校準)"""
        raw_score = np.dot(self.w, x) + self.b
        if self.calibrator:
            return self.calibrator.predict_proba([[raw_score]])[0][1]
        return 1 / (1 + np.exp(-raw_score))

# ========== 3. 訓練流程 ==========
def load_dataset(pos_dir, neg_dir, num_neg_samples=1000):
    """加載數據集並提取特徵"""
    pos_images = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]
    neg_images = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]
    
    # 提取正樣本特徵
    pos_feats = []
    for path in tqdm(pos_images, desc="正樣本"):
        feat = extract_hog_features(path)
        if feat is not None:
            pos_feats.append(feat)
    
    # 提取負樣本特徵 
    neg_feats = []
    for path in tqdm(random.sample(neg_images, num_neg_samples), desc="負樣本"):
        feat = extract_hog_features(path)
        if feat is not None:
            neg_feats.append(feat)
    
    return np.array(pos_feats), np.array(neg_feats)

def train_ensemble(pos_feats, neg_feats):
    """訓練Exemplar-SVM集成模型"""
    models = []
    for i, x_E in enumerate(tqdm(pos_feats, desc="訓練Exemplar-SVMs")):
        svm = ExemplarSVM(C1=0.5, C2=0.01)
        svm.fit(x_E, neg_feats)
        models.append(svm)
    return models

# ========== 4. 滑動窗口檢測 ==========
def sliding_window_detect(image_path, models, window_size=(128,128), step_size=32, threshold=0.5):
#滑動窗口檢測 (帶可視化)
    image = rgb2gray(imread(image_path))
    image = resize(image, (256, 256), anti_aliasing=True)
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    
    candidate_boxes = []
    
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            window = image[y:y+window_size[1], x:x+window_size[0]]
            
            if window.shape != window_size:
                continue
                
            feat, _ = hog(window,
                          pixels_per_cell=(8,8),
                          cells_per_block=(2,2),
                          orientations=9,
                          block_norm='L2-Hys',
                          visualize=True)
            
            # 集成預測 (論文第3節)
            scores = [model.predict(feat) for model in models]
            avg_score = np.mean(scores)
            
            if avg_score >= threshold:
                candidate_boxes.append((x, y, avg_score))
                rect = plt.Rectangle((x, y), window_size[0], window_size[1], 
                                    linewidth=1, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
    
    # 標記最佳檢測結果
    if candidate_boxes:
        best_box = max(candidate_boxes, key=lambda x: x[2])
        x, y, score = best_box
        rect = plt.Rectangle((x, y), window_size[0], window_size[1], 
                            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Best Detection (Score: {score:.3f})")
    
    plt.axis('off')
    plt.show()
#%%
# ========== 主流程 ==========
if __name__ == "__main__":
    # 1. 加載數據
    pos_feats, neg_feats = load_dataset(
        pos_dir='sample/train/po/',
        neg_dir='sample/train/ne/',
        num_neg_samples=4
    )
    
    # 2. 訓練模型集成
    models = train_ensemble(pos_feats, neg_feats)
    
    # 3. 測試檢測
    sliding_window_detect(
        image_path="sample/test/7.jpg",
        models=models,
        window_size=(128,128),
        step_size=32,
        threshold=0.4
    )