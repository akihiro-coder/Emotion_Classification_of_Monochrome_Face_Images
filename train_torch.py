import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import glob
import os
from PIL import Image



# set machine resource
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# define preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])


# prepare dataset
dataset = datasets.ImageFolder('./data/train/label', transform=transform)

# split dataset to train data and validation data
n_samples = len(dataset)
train_size = int(len(dataset) * 0.8)
val_size = n_samples - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# prepare dataloader
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# label name
emotion_name = ('sad', 'angry', 'neutral', 'happy')
num_classes = len(emotion_name)

# define model structure
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 特徴抽出層
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(in_features=8 * 8 * 128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# creating an object for CNN class
model = CNN(num_classes=num_classes)
model.to(device)  # on GPU

# define criterion
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# lists of saving loss values and accuracy values
losses = []
accs = []
val_losses = []
val_accs = []


# train loop
num_epochs = 3
for epoch in range(num_epochs):
    # train loop
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in train_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        running_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        optimizer.step()
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)

    # validation loop
    val_running_loss = 0.0
    val_running_acc = 0.0
    for val_imgs, val_labels in val_dataloader:
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        val_output = model(val_imgs)
        val_loss = criterion(val_output, val_labels)
        val_running_loss += val_loss.item()
        val_pred = torch.argmax(val_output, dim=1)
        val_running_acc += torch.mean(val_pred.eq(val_labels).float())
    val_running_loss /= len(val_dataloader)
    val_running_acc /= len(val_dataloader)
    val_losses.append(val_running_loss)
    val_accs.append(val_running_acc)
    print(f'epoch: {epoch}, loss: {running_loss}, acc: {running_acc}, val_loss: {val_running_loss}, val_acc: {val_running_acc}')


## graph result →　function
#x = [x for x in range(len(losses))]  # x axis
#accs_list = []
#for acc in accs:
#    acc = acc.item()  # get float variable from torch.tensor variable
#    accs_list.append(acc)
#
#val_accs_list = []
#for acc in val_accs:
#    acc = acc.item()  # get float variable from torch.tensor variable
#    val_accs_list.append(acc)
#
#plt.plot(x, losses, label='loss', color='b')
#plt.plot(x, val_losses, label='val_loss', color='c')
#plt.legend()
#plt.savefig('./logs/pytorch/loss.png')
#plt.close()
#
#plt.plot(x, accs_list, label='acc', color='r')
#plt.plot(x, val_accs_list, label='val_acc', color='m')
#plt.legend()
#plt.savefig('./logs/pytorch/acc.png')
#plt.close()
#print('save graph finished')



# save model
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)

# load model
model.load_state_dict(torch.load(model_path))
model.eval()

# make test images to torch.tensor
#ファイル数
#ファイル数だけ読み込みループ
#    read image
#    前処理
#    推論
#    csvファイルに書き込む
#
#file_length = 
data_dir = './data/test'
search_pattern = '*.jpg'
image_path_list = glob.glob(os.path.join(data_dir, search_pattern))
# preprocessing
trainsform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((128, 128))
    ])
images = []
for image_path in image_path_list:
    img = Image.open(image_path)
    img_tensor = transform(img)
    output = model(img_tensor)
    print(output)
    pred = torch.argmax(output, dim=1)
    print(pred)
    exit()
    


