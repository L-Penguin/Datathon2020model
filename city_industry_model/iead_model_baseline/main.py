from dataloader import load_data
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def main():
    device = torch.device('cuda:0')

    para_list, iead_list = load_data()

    model = nn.Sequential(
        nn.Linear(4, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        # 6 hidden layers
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        # nn.Linear(2**14, 2**12, bias=False),
        # nn.ReLU(inplace=False),
        # nn.Linear(2**16, 2**14, bias=False),
        # nn.ReLU(inplace=False),
        # nn.Linear(2**16, 2**16, bias=False),
        # nn.ReLU(inplace=False),
        # end
        nn.Linear(512, 500*180)
    )

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss(reduction='mean')

    losses = []

    model.train()
    for epoch in range(0, 3000):
        print("training ", epoch)
        for idx, each_data in enumerate(iead_list):
            
            X = Variable(torch.FloatTensor(para_list[idx]), requires_grad = True).to(device)
            Y = Variable(torch.FloatTensor(iead_list[idx]), requires_grad = False).to(device)

            y_pred = model(X)

            loss = loss_func(Y, y_pred)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    torch.save(model, 'model.pth')

    model.eval()
    for idx, paras in enumerate(para_list):
        parameter_dir = './out/[' + str(paras[0]) + ',' + str(paras[1]) + ',' + str(paras[2]) + str(paras[3]) + ']'
        prediction = model(Variable(torch.FloatTensor(paras), requires_grad = True).to(device)).to(device)
        prediction = prediction.data.cpu().numpy()
        fig = plt.imshow(prediction.reshape(500, 180))
        plt.savefig(parameter_dir + 'pred.png')
        fig = plt.imshow(iead_list[idx].reshape(500, 180))
        plt.savefig(parameter_dir + 'actual.png')
        plt.close()

if __name__ == "__main__":
    main()
