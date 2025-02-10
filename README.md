---
tags: Aidea competition
---

# AOI 瑕疵分類


## 比賽連結
https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4

## 簡介
本專案目的為藉由AOI影像訓練深度學習模型辨識產品表面瑕疵。結果顯示，訓練後的預訓練DenseNet121模型的測試準確達到98.89%、ResNet50模型達到97.75%及EfficientNet-B0模型98.96%，最後我們透過投票的方式綜合三個模型的結果為最後的預測結果，綜合後的集成模型準確達到99.21%。(目前排行榜上最高分為99.8%) 未來有時間會再嘗試更大的模型架構(如DenseNet169、EfficientNet-B4)，相信能進一步提升測試準確率。

## 資料說明
訓練資料：2,528張(隨機抽取20%作為驗證資料)
測試資料：10,142張
影像類別：6 個類別(正常類別 + 5種瑕疵類別)
影像尺寸：512x512


## 瑕疵分類
如下圖所示，除了Normal外，其餘皆屬於瑕疵影像。

![image](https://github.com/andylee40/aoi_defect_detection/blob/main/dataset.png)


## 資料不平衡處理
下圖為訓練資料集中的標籤分佈，可看出在第二類與的資料比較少，存在資料不平衡的問題。因此使用Focal Loss作為本次所有模型使用的損失函數，並針對訓練資料集中的標籤分佈，給予不同標籤相異的權重（越多的標籤給予越小的權重，越少的標籤給予越大的權重），以增強模型對少數標籤樣本的判別能力。
![image](https://github.com/andylee40/aoi_defect_detection/blob/main/label.png)


## 影像增強
為增強模型提取影像特徵能力，針對所有影像進行以下影像預處理與增強：
* 高斯模糊
* 自適應直方圖均衡化
* Laplacian銳化

下圖為影像增強前後對比，可發現影像增強後，影像細節更明顯。

![image](https://github.com/andylee40/aoi_defect_detection/blob/main/enhanced.png)


## 影像擴增
* 影像隨機水平翻轉(p=0.5)
* 影像隨機旋轉正負 15 度
* 影像大小縮放成 224 x 224

## 模型
* DenseNet121 (pretrained)
* ResNet50 (pretrained)
* EfficientNet-B0 (pretrained)


## 模型訓練設置
我們所有模型訓練設置如下，當驗證損失訓練5次後未下降，調降學習率;當驗證損失訓練7次後未下降，停止訓練。儲存驗證損失最低的模型權重，當做後續預測測試資料之模型。
* Epoch：50
* Learning rate：0.001
* Weight decay：0.00001
* Optimizer：AdamW
* Scheduler：ReduceLROnPlateau(patience=5)
* Early stopping patience： 7 

## 結果
1. 下表為預測結果，原以為三個模型中最大的模型ResNet50預測結果會最好，但就單一模型來看，可看出以預訓練EfficientNet-B0輸入AOI影像訓練後的辨識結果最佳，對10,142張測試資料的準確度(Accuracy)已達到98.96%。
1. 最終使用的集成投票預測，可以達到更好的準確率99.21%。
1. 註：測試資料的準確度是將預測結果上傳Aidea平台，由Aidea平台評分而得。

| 模型 | 訓練準確率 | 驗證準確率 |測試準確率 |
|:------|:------:|:------:|:------:|
| DenseNet121 | 99.8%  | 99.01% |98.89% |
| ResNet50 | 99.1%  | 97.82% |97.75% |
| EfficientNet-B0 | 99.01% | 98.61% |98.96% |
| VOTE | - | - |99.21% |





## 排行榜

最終最優排名為100名，參賽人數有834組，排名為前12%。

![image](https://github.com/andylee40/aoi_defect_detection/blob/main/leaderboard.png)