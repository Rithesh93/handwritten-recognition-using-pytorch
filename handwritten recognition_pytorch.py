import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def image_show(model):
    plt.imshow(model[0].numpy().reshape(28,28), cmap='gray')
    plt.xlabel('y = ' +str(model[1]))
    plt.show()
    
    
""" training and validation set for datasets in MNIST"""   
training_set = dsets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
validation_set = dsets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())

"""display an image for a particular class"""
image_show(training_set[70])

""" plot images of 10 classes (0-9) with 2 rows and 5 columns"""
def modelparameters(model):
    W = model.state_dict()['linear.weight'].data
    W_min = W.min().item()
    W_max = W.max().item()
    fig, axes = plt.subplots(2,5)
    for i,ax in enumerate(axes.flat):
        if i<10:
            ax.set_xlabel('class : {0}'.format(i))
            ax.imshow(W[i,:].view(28,28), vmin=W_min, vmax=W_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()
    
"""Softmax class"""
class Softmax(nn.Module):
    def __init__(self,in_size,out_size):
        super(Softmax,self).__init__()
        self.linear=nn.Linear(in_size,out_size)
    def forward(self,x):
        return self.linear(x)


in_size=28*28
out_size=10
model = Softmax(in_size,out_size)   #computes the model
criterion = nn.CrossEntropyLoss()   #computes the softmax function as well as the cross-entropy loss
lr=0.01  #learning_rate

optim = torch.optim.SGD(model.parameters(), lr=lr)

modelparameters(model)

trainloader = DataLoader(dataset=training_set, batch_size=100)
validationloader = DataLoader(dataset=validation_set, batch_size=5000)
n_epochs=10
N = len(validation_set)

#train the model
for epoch in range(n_epochs):
    for x,y in trainloader:
        yhat = model(x.view(-1,28*28))
        loss = criterion(yhat,y)
        loss.backward()
        optim.step()
    correct=0
#test the model 
    for x_test, y_test in validationloader:
        z = model(x_test.view(-1,28*28))
        _,yhat_test=torch.max(z.data,1)
        #correct+=(yhat==yhat_test).sum().item()
        #accuracy = correct/N

modelparameters(model)

#prediction
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_set:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        image_show((x, y))
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break



