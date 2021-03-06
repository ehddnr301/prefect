from prefect import task
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
# from utils import MnistDataset, MnistNet
from sklearn.neighbors import KNeighborsClassifier
from abc import *
# import matplotlib.pyplot as plt
from prefect import Task, Flow
import requests

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
def cnn_training(self, Net, train_loader, epoch, device, criterion, optimizer, total_batch):
    for epoch in range(epoch):
        avg_cost = 0

        for X, Y in train_loader: # ?????? ?????? ????????? ????????????. X??? ?????? ??????, Y??? ????????????.
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

@task
def load_dataset(self, df_path):
    df = pd.read_csv(df_path)
    train_df, valid_df = train_test_split(df, test_size=0.1, stratify=df['label'])

    return train_df, valid_df

@task
def preprocess_train(self, Dataset, transform, batch_size, train_df, valid_df):
    trainset = Dataset(train_df, transform)
    validset = Dataset(train_df, transform)
    train_loader = DataLoader(trainset, batch_size=batch_size)
    valid_loader = DataLoader(validset, batch_size=batch_size)
    total_batch = len(train_loader)

    return train_loader, valid_loader, total_batch

@task
def save_model(self):
    print('model ????????? ???????????? save!')

@task
def make_knn_feature(self, train_df, train_loader, device, optimizer, Net2):
    temp = pd.DataFrame(columns=[f'{i}_feature' for i in range(32)], index=train_df.index)
        # 32 ??? ???????????????????????? ???????????? feature ??????
    batch_index = 0
    batch_size = train_loader.batch_size

    for i,(mini_batch, _) in enumerate(train_loader): # ?????? ?????? ????????? ????????????.
        mini_batch = mini_batch.to(device)
        optimizer.zero_grad()
        outputs = Net2(mini_batch)
        batch_index = i * batch_size
        temp.iloc[batch_index:batch_index+batch_size,:] = outputs.detach().numpy()

    temp.reset_index(inplace=True)
    feature_weight_df = temp

    return feature_weight_df

@task
def knn_training(self, neighbors, feature_weight_df):
    KNN = KNeighborsClassifier(n_neighbors=neighbors)
    KNN.fit(feature_weight_df.iloc[:, 1:].values, feature_weight_df.iloc[:,0].values)

    return KNN

@task
def predict_knn_model(self, num, train_df, valid_df, tranform, Net2, KNN):
    test_mnist = valid_df.iloc[num,1:] # ????????????????????? parameter??? ????????? ??? ???
    test_data = tranform(test_mnist.values.astype(np.uint8).reshape((28, 28))).unsqueeze(0)
    knn_result = Net2(test_data).detach().numpy()
    num = KNN.predict(knn_result)[0]
    knn_result = train_df[train_df.index == num].iloc[0,1:].values.reshape(28,28)

    return knn_result

def train_cnn_model(self):
    DF_PATH = './train.csv'
    BATCH_SIZE = 64
    TRAINING_EPOCH=3
    DEVICE = 'cpu'
    CRITERION = torch.nn.CrossEntropyLoss().to(DEVICE)
    LEARNING_RATE = 1e-3
    Net = MnistNet()
    optimizer = torch.optim.Adam(Net.parameters(), lr=LEARNING_RATE)

    a = load_dataset(df_path=DF_PATH)
    train_df, valid_df = a
    train_loader, valid_loader, total_batch = preprocess_train(Dataset=Dataset, transform=train_transform, batch_size=BATCH_SIZE, train_df=train_df, valid_df=valid_df)
    train_Net = cnn_training(Net=Net, train_loader=train_loader, epoch=TRAINING_EPOCH, device=DEVICE, criterion=CRITERION, optimizer=optimizer, total_batch=total_batch)
    save_model(train_Net)
    Net2 = torch.nn.Sequential(*list(train_Net.children())[:-1])
    feature_weight_df = make_knn_feature(train_df=train_df, train_loader=train_loader, device=DEVICE, optimizer=optimizer, Net2=Net2)
    KNN = knn_training(neighbors=3, feature_weight_df=feature_weight_df)
    knn_result = predict_knn_model(323, train_df=train_df, valid_df=valid_df, transform=train_transform, Net2=Net2, KNN=KNN)

    


    # def show_result(self):
    #     plt.figure()

    #     #subplot(r,c) provide the no. of rows and columns
    #     f, axarr = plt.subplots(2,1) 

    #     # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    #     axarr[0].imshow(self.test_mnist.values.astype(np.uint8).reshape((28, 28)))
    #     axarr[1].imshow(self.knn_result)


if __name__ == '__main__':

    mnist = MnistPrefects(
        df_path='./train.csv',
        Dataset=MnistDataset,
        Net=MnistNet,
        learning_rate=1e-3,
        training_epochs=3,
        batch_size=64,
        neighbors=3
    )

    mnist.run()




