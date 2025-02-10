import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import cv2
import copy  
import os
import matplotlib.pyplot as plt


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # self.alpha = alpha
        self.alpha = None if alpha is None else torch.tensor(alpha)

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[labels] 
            focal_weight *= alpha_t

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
    
def train_model(n_epochs, training_loader, validation_loader, model,device, criterion,
                optimizer, scheduler,history,mode):

    best_model_wts = copy.deepcopy(model.state_dict())
    valid_loss_min = np.Inf
    early_stopper = EarlyStopper(patience=7, min_delta=0)

    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        valid_loss = 0
        train_labels = []
        train_probs = []
        valid_labels = []
        valid_probs = []
        train_preds=[]
        valid_preds=[]

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))

        for batch_idx, data in enumerate(training_loader):
            image = data['image'].to(device)
            label = data['label'].long().to(device)
            prob_vec, preds= model(image)
            prob_vec = prob_vec.to(device)
            loss = criterion(prob_vec, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * label.size(0)
            
            train_labels.extend(label.cpu().numpy())
            train_probs.extend(prob_vec[:, 1].detach().cpu().numpy())
            train_preds.extend(preds.cpu().numpy())

        print('############# Epoch {}: Training End     #############'.format(epoch))
        print('############# Epoch {}: Validation Start   #############'.format(epoch))

        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                image = data['image'].to(device)
                label = data['label'].long().to(device)
                prob_vec, preds= model(image)
                prob_vec = prob_vec.to(device)
                loss = criterion(prob_vec, label)
                valid_loss += loss.item() * label.size(0)

                valid_labels.extend(label.cpu().numpy())
                valid_probs.extend(prob_vec[:, 1].detach().cpu().numpy())
                valid_preds.extend(preds.cpu().numpy())


        print('############# Epoch {}: Validation End     #############'.format(epoch))
        
        # 計算平均損失
        train_loss = train_loss / len(training_loader.sampler)
        valid_loss = valid_loss / len(validation_loader.sampler)

        # 計算 ACCURACY 分數
        train_correct = sum(p == l for p, l in zip(train_preds, train_labels))
        valid_correct = sum(p == l for p, l in zip(valid_preds, valid_labels))
        
        train_accu = train_correct/ len(training_loader.sampler)
        valid_accu = valid_correct/ len(validation_loader.sampler)

        history['epoch'].append(epoch)
        history['train_accu'].append(train_accu)
        history['train_loss'].append(train_loss)
        history['valid_accu'].append(valid_accu)
        history['valid_loss'].append(valid_loss)

        print('Epoch: {} \tAverage Training Loss: {:.6f} \tTraining ACC: {:.6f}  \tAverage Validation Loss: {:.6f} \tValidation ACC: {:.6f} '.format(
            epoch, 
            train_loss,
            train_accu,
            valid_loss,
            valid_accu,
        ))


        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': valid_loss,
            'valid_accuracy_max': valid_accu,
            'state_dict': model.state_dict(),
        }
        scheduler.step(valid_loss)
        # scheduler.step()
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(checkpoint, '.weights/{}_best.pt'.format(mode))

        if early_stopper.early_stop(valid_loss):
            break
        print('############# Epoch {}  Done   #############\n'.format(epoch))

    model.load_state_dict(best_model_wts)
    # return model
    
    
    
def test(model,device, test_loader):
    # 儲存資料
    test_targets = []
    test_outputs = []
    test_embedd = []
    test_provec = []

    test_accu = 0
    test_loss = 0

    model.eval()
    # 評估的時候不需要更新參數、計算梯度
    with torch.no_grad():
        # 驗證集小批量迭代
        for batch_idx, data in enumerate(test_loader, 0):
            image = data['image'].to(device)
            prob_vec, preds= model(image)
            test_outputs.extend(preds.cpu().numpy())
    return test_outputs



def plot_result(history):

    plt.figure(figsize=(8,6))
    plt.plot(history['epoch'], history['train_loss'], label='train_loss', marker="o")
    plt.plot(history['epoch'], history['valid_loss'], label='valid_loss', marker="*")
    plt.title('Training loss history', fontsize=16)
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(history['epoch'], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()
    
    
    plt.figure(figsize=(8,6))
    plt.plot(history['epoch'], history['train_accu'], label='train_accu', marker="o")
    plt.plot(history['epoch'], history['valid_accu'], label='valid_accu', marker="*")
    plt.title('Training accu history', fontsize=16)
    plt.ylabel('Accu', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(history['epoch'], rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()
    

def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    valid_loss_min = checkpoint.get('valid_loss_min', None)
    valid_accuracy_max = checkpoint.get('valid_accuracy_max', None)
    print(f"✅ 成功載入模型，來自 epoch: {checkpoint['epoch']}")
    
    if valid_loss_min is not None and valid_accuracy_max is not None:
        print(f"valid_loss_min: {valid_loss_min:.6f}，valid_accuracy_max: {valid_accuracy_max:.6f}")
    elif valid_loss_min is not None:
        print(f"valid_loss_min: {valid_loss_min:.6f}")
    else:
        print("checkpoint 沒有資訊")

    return model


def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # 計算參數大小
    buffer_size = sum(p.numel() * p.element_size() for p in model.buffers())  # 計算 buffer 大小
    total_size = (param_size + buffer_size) / 1024**2  # 轉換為 MB
    print(f"Model Size: {total_size:.2f} MB")
    # return total_size