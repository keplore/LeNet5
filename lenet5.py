# -*-coding:utf-8-*-
# date   : 2020-1-27
# addr   : HangZhou China
# author : yangshuo
# notes  : lenet-5 pytorch code
# command: python lenet5.py
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse

model_path = './model/'
if not os.path.exists(model_path):
    os.mkdir(model_path)

seed = 2020
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# define CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# define LeNet5
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(               # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),             # padding=2 input == output
            nn.ReLU(),                            # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),# output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),          # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # define forward backward network，input x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten to 1 dimension
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') # model path
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")     # model dir
opt = parser.parse_args()

# hyper parameters setting
EPOCH      = 10      # epoch
BATCH_SIZE = 64      # batch_size
LR         = 0.001   # learning rate

# preprocessing method
transform = transforms.ToTensor()

# define train datasets
trainset = tv.datasets.MNIST(root='./data/',
                             train=True,
                             download=True,
                             transform=transform)

# define train batch datasets
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          )

# define test datasets
testset = tv.datasets.MNIST(root='./data/',
                            train=False,
                            download=True,
                            transform=transform)

# define test batch datasets
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         )

# define loss function and optimization method
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))

# main
if __name__ == "__main__":
    cur_acc = 0.0
    for epoch in range(EPOCH):
        sum_loss = 0.0
        # data loading
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # gradient reset
            optimizer.zero_grad()

            # forward and backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # every 100 batch print average loss once
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.012f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # each epoch test accuracy once
        with torch.no_grad():
            correct = 0
            total   = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # max score
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('epoch %d test accuracy：%.012f%%' % (epoch + 1, (100.0 * correct / total)))
            if(cur_acc < (100.0 * correct / total)):
                cur_acc = (100.0 * correct / total)
                torch.save(net.state_dict(), '%sbest.pth' % (opt.outf))
    torch.save(net.state_dict(), '%snet_%d.pth' % (opt.outf, epoch + 1))

