import torch
import torch.nn as nn
import torch.optim as optim
import argparse

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

import yaml
import os
from loguru import logger


def load_hparams(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        hparams = yaml.safe_load(f)
    return hparams


def save_hparams(hparams, log_dir):
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(hparams, f)


batch_size = 100
learning_rate = 0.0002
num_epoch = 10
log_dir = './logs'

mnist_train = datasets.MNIST(root="../Deep_learn/Data/", train=True, transform=transforms.ToTensor(),
                             target_transform=None, download=True)
mnist_test = datasets.MNIST(root="../Deep_learn/Data/", train=False, transform=transforms.ToTensor(),
                            target_transform=None, download=True)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        # super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()

        # batch_size = 100
        self.layer = nn.Sequential(
            # [100,1,28,28] -> [100,16,24,24]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),

            # [100,16,24,24] -> [100,32,20,20]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),

            # [100,32,20,20] -> [100,32,10,10]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [100,32,10,10] -> [100,64,6,6]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),

            # [100,64,6,6] -> [100,64,3,3]
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            # [100,64*3*3] -> [100,100]
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            # [100,100] -> [100,10]
            nn.Linear(100, 10)
        )

    def forward(self, x):
        # self.layer에 정의한 연산 수행
        out = self.layer(x)
        # view 함수를 이용해 텐서의 형태를 [100,나머지]로 변환
        out = out.view(batch_size, -1)
        # self.fc_layer 정의한 연산 수행
        out = self.fc_layer(out)
        return out


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.ckpt'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved to {filename}')


def load_checkpoint(model, optimizer, filename='checkpoint.ckpt'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded from {filename}')
    return start_epoch, loss


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
parser.add_argument('--cfg', type=str, default='config.yaml', help='path to configuration file')
args = parser.parse_args()

hparams = load_hparams(args.cfg)
batch_size = hparams.get('batch_size', 100)
learning_rate = hparams.get('learning_rate', 0.0002)
num_epoch = hparams.get('num_epoch', 10)
log_dir = hparams.get('log_dir', './logs')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

start_epoch = 0
if args.resume:
    start_epoch, _ = load_checkpoint(model, optimizer, args.resume)

loss_arr = []
loss_func = nn.CrossEntropyLoss()

for epoch in range(start_epoch, num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y = label.to(device)

        optimizer.zero_grad()

        output = model.forward(x)

        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

        if j % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}], Step [{j}/{len(train_loader)}], Loss: {loss.item()}")
            logger.info(f"Epoch [{epoch + 1}/{num_epoch}], Step [{j}/{len(train_loader)}], Loss: {loss.item()}")
            loss_arr.append(loss.cpu().detach().numpy())

    save_checkpoint(model, optimizer, epoch, loss.item())

correct = 0
total = 0
y_true = []
y_pred = []

# evaluate model
model.eval()

with torch.no_grad():
    for image, label in test_loader:
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)

        # torch.max함수는 (최댓값,index)를 반환
        _, output_index = torch.max(output, 1)

        # 전체 개수 += 라벨의 개수
        total += label.size(0)

        # 도출한 모델의 index와 라벨이 일치하면 correct에 개수 추가
        correct += (output_index == y).sum().float()

        y_true.extend(y.cpu().numpy())
        y_pred.extend(output_index.cpu().numpy())

    logger.info("Accuracy of Test Data: {}%".format(100 * correct / total))

    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100 * correct / total))

# 혼동 행렬 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

save_hparams(hparams, log_dir)
