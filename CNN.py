import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
learning_rate = 0.0002
num_epoch = 10

mnist_train = datasets.MNIST(root="../Data/", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = datasets.MNIST(root="../Data/", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []
for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y = label.to(device)

        optimizer.zero_grad()

        output = model.forward(x)

        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

        if j % 1000 == 0:
            print("loss",loss)
            loss_arr.append(loss.cpu().detach().numpy())

correct = 0
total = 0

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

    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100 * correct / total))