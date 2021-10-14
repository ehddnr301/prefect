import prefect
from prefect import task, Parameter


@task
def hello_task(name):
    logger = prefect.context.get("logger")
    logger.info(f"Hello {name}!")

    return name


@task
def hi_task(name):
    logger = prefect.context.get("logger")
    logger.info(f"Hi {name}!")

    return name


@task
def buy_task(name):
    logger = prefect.context.get("logger")
    logger.info(f"Bye {name}!")




from prefect import task
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
# from utils import MnistDataset, MnistNet
from sklearn.neighbors import KNeighborsClassifier
from abc import *
# import matplotlib.pyplot as plt

class MnistDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
                
        image = item[1:].values.astype(np.uint8).reshape((28, 28))
        label = item[0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

class MnistNet(torch.nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.flatten = torch.nn.Flatten()

        self.fc = torch.nn.Linear(7 * 7 * 64, 32, bias=True)
        self.last_layer = torch.nn.Linear(32, 10, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.last_layer(out)
        return out
train_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(1.0,))
                        ])





@task
def cnn_training( Net, train_loader, epoch, device, criterion, optimizer, total_batch):
    for epoch in range(epoch):
        avg_cost = 0

        for X, Y in train_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = Net(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    
    return Net

@task(nout=2)
def load_dataset(df_path):
    df = pd.read_csv(df_path)
    train_df, valid_df = train_test_split(df, test_size=0.1, stratify=df['label'])

    return (train_df, valid_df)

@task(nout=3)
def preprocess_train( Dataset, transform, batch_size, train_df, valid_df):
    trainset = Dataset(train_df, transform)
    validset = Dataset(train_df, transform)
    train_loader = DataLoader(trainset, batch_size=batch_size)
    valid_loader = DataLoader(validset, batch_size=batch_size)
    total_batch = len(train_loader)

    return (train_loader, valid_loader, total_batch)

@task
def save_model():
    print('model 성능이 괜찮아서 save!')

@task
def make_knn_feature( train_df, train_loader, device, optimizer, Net2):
    temp = pd.DataFrame(columns=[f'{i}_feature' for i in range(32)], index=train_df.index)
        # 32 는 마지막레이어에서 생성하는 feature 갯수
    batch_index = 0
    batch_size = train_loader.batch_size

    for i,(mini_batch, _) in enumerate(train_loader): # 미니 배치 단위로 꺼내온다.
        mini_batch = mini_batch.to(device)
        optimizer.zero_grad()
        outputs = Net2(mini_batch)
        batch_index = i * batch_size
        temp.iloc[batch_index:batch_index+batch_size,:] = outputs.detach().numpy()

    temp.reset_index(inplace=True)
    feature_weight_df = temp

    return feature_weight_df

@task
def knn_training( neighbors, feature_weight_df):
    KNN = KNeighborsClassifier(n_neighbors=neighbors)
    KNN.fit(feature_weight_df.iloc[:, 1:].values, feature_weight_df.iloc[:,0].values)

    return KNN

@task
def predict_knn_model( num, train_df, valid_df, transform, Net2, KNN):
    test_mnist = valid_df.iloc[num,1:] # 실제동작에서는 parameter로 받아야 할 값
    test_data = transform(test_mnist.values.astype(np.uint8).reshape((28, 28))).unsqueeze(0)
    knn_result = Net2(test_data).detach().numpy()
    num = KNN.predict(knn_result)[0]
    knn_result = train_df[train_df.index == num].iloc[0,1:].values.reshape(28,28)

    return knn_result

@task
def return_Net2(Net):
    return torch.nn.Sequential(*list(Net.children())[:-1])