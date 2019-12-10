from ConvNet import convnet
import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm.autonotebook import tqdm

device = torch.device('cpu')
print_epoch = 5
path_result = "results"
### Dataset
# Use standard FashionMNIST dataset
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.], [0.5])])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

### Model
model = convnet.Net().to(device)
model.apply(convnet.Net.init_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
lossCE = torch.nn.CrossEntropyLoss()
################################################
# TRAIN 
################################################
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress = tqdm(enumerate(trainloader), desc="Train", total=len(trainloader))
    for x in progress:
        
        data    = x[1]
        image   = data[0].to(device)
        label   = data[1].to(device)
        
        
        optimizer.zero_grad()
        out_class = model(image)
        loss_ce    = lossCE(out_class, label)
        loss_total = loss_ce
        loss_total.backward()
        optimizer.step()

        train_loss += loss_total.item()
        # Calculate accuracy
        _, predicted = torch.max(out_class, 1)
        correct += (predicted == label).float().sum()
        total += len(predicted)
        progress.set_description("Train epoch: %d, CE: %.5f Acc.: %.3f" % (epoch, loss_total.item(), (correct*100)/total))
    
    # Save model
    if (epoch + 1) % print_epoch == 0:

        # Save the model
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
        torch.save(state, path_result + "/checkpoint" + str(epoch) + ".pth.tar" ) 

    return train_loss

################################################
# TEST 
################################################
def test(epoch):
    model.eval()
    test_loss   = 0
    image       = None
    image_synt  = None
    correct = 0
    total = 0
    progress = tqdm(enumerate(testloader), desc="Test", total=len(testloader))
    with torch.no_grad():
        count_total = 0
        count_true  = 0
        for x in progress:
          data    = x[1]
          image   = data[0].to(device)
          label   = data[1].to(device)

          out_class = model(image)
          loss_mse    = lossCE(out_class, label)
          loss_total =  loss_mse
            
          test_loss += loss_total.item()

          results     = {}
        
          _, predicted = torch.max(out_class, 1)
          correct += (predicted == label).float().sum()
          total += len(predicted)
          progress.set_description("Test epoch: %d, CE: %.5f , Acc.: %.3f" % (epoch, loss_total.item(), (correct*100)/total))
        # print("Example ")

        # porcentaje = (count_true*100)/count_total
        # print("Rank-1: ", round(porcentaje, 3))
       
    return test_loss

best_test_loss = float('inf')
patience_counter = 0

loss_Train = []
loss_Test = []
num_epochs = 1000

  
for e in range(num_epochs):

    train_loss = train(e)
    test_loss = test(e)

    train_loss /= len(trainloader)
    test_loss /= len(testloader)
    loss_Train.append(train_loss)
    loss_Test.append(test_loss)

    if (e + 1) % print_epoch == 0:
      #plot 
      fig = plt.figure()
      x = np.arange(len(loss_Train))
      ax = plt.subplot(111)
      ax.plot(x, np.array(loss_Train), 'mediumvioletred', label='Generator Training')
      ax.plot(x, np.array(loss_Test), 'pink', label='Generator Test')

      plt.title('Function loss')
      ax.legend()
      fig.savefig(path_result + '/plot' + str(num_epochs+1) + '.png')

      # Save loss an test values to plot comparison
      fichero = open(path_result + '/files_train.pckl', 'wb')
      pickle.dump(loss_Train, fichero)
      fichero.close()
      fichero = open(path_result + '/files_test.pckl', 'wb')
      pickle.dump(loss_Test, fichero)
      fichero.close()

