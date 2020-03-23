import torch
import torch.functional as F
import torch.nn as nn

import time
import numpy as np

class dataset(torch.utils.data.Dataset):
    '''
    Loader for the pyotorch models.
    '''
    def __init__(self,X_train,Y_train, cuda=False):
        '''
        Expects numpy arrays
        '''
        self.X_train = torch.from_numpy(X_train)
        self.Y_train = torch.from_numpy(Y_train).long()
        '''if cuda:
            self.X_train = self.X_train.cuda()
            self.Y_train = self.Y_train.cuda()'''

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]
    
    def __len__(self):
        return len(self.X_train)



class Egg_nn(nn.Module):
    def __init__(self, n_channels=7, n_samples=500):
        super().__init__()
        self.n_channels, self.n_samples = n_channels, n_samples
        # Compute the size of the last fully connected layer
        w_init = n_samples
        w = w_init
        for i in range(4):
            w = w-1 # Conv
            w = w // 2 # Pooling
        for i in range(2):
            w = w-1
        self.n_hid = n_channels * w
        # TODO: add delated conv 
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 50, (1,2)),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(50, 50, (1,2)),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(50, 20, (1,2)),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(20, 20, (1,2)),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(20, 20, (1,2)),
            nn.ReLU(),
            nn.Conv2d(20, 20, (1,2)),
            nn.ReLU(),
            )
        self.fc = nn.Linear(self.n_hid*20,2)

    def forward(self, input):
        a,b,c = input.shape
        input = input.view(a, 1, b, c)
        x = self.convolutions(input)
        x = x.view(-1, self.n_hid*20)
        scores = self.fc(x)
        return scores



class Egg_module():
    def __init__(self, lr=0.002, criterion=nn.CrossEntropyLoss(), cuda=False):
        self.model = Egg_nn()
        if cuda: self.model = self.model.cuda()
        self.cuda = cuda
        self.lr = lr
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def train(self, datasetTrain, batch_size, epochs, shuffle = True, test = None):
        train_loader = torch.utils.data.DataLoader(datasetTrain,
        batch_size=batch_size, shuffle=shuffle, num_workers=1)

        val_loader = torch.utils.data.DataLoader(test, batch_size = 75, num_workers =1)

        for epoch in range(epochs):
            start = time.time()
            self.model.train()
            list_loss = []
            for x,y in train_loader:
                if self.cuda: x,y = x.cuda(), y.cuda()
                output = self.model(x)
                loss = self.criterion(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                list_loss.append(loss.detach().cpu().numpy())

            print(f"Epoch {epoch} : Loss Total {np.stack(list_loss).mean()}, time {time.time()-start}")
            mid = time.time()

            self.model.eval()
            list_loss = []
            total = 0
            correct = 0
            for x,y in val_loader:
                #print(x.shape, y.shape)
                if self.cuda: x,y = x.cuda(), y.cuda()
                output = self.model(x)
                #print(output.shape)
                loss = self.criterion(output,y)
                list_loss.append(loss.detach().cpu().numpy())
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).cpu().sum().item()

            print('Validation accuracy: %d %%, time %f' % (100 * correct / total, time.time()-mid))

        def predict(self, x_eval):
            y_eval = self.model(x_eval)
            return y_eval
    
        def save_weight(self,name_input = "model_weights.pth"):
            state_dict = self.model.state_dict()
            torch.save(state_dict, name_input)

        def load(self,name_weights='model_weights.pth'):
            state_dict = torch.load(name_weights)
            self.model.load_state_dict(state_dict)

