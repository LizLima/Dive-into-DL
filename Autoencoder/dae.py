import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from skimage.util import random_noise
import matplotlib.pyplot as plt


import torch

latent_dims = 10
num_epochs = 50
batch_size = 128
capacity = 64
learning_rate = 1e-3
device = torch.device('cuda')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x)) # last layer before output is tanh, since the images are normalized and 0-centered
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


def load_data():

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def add_noise(imgs, noise_type, mean=0, var=0.05):
  # Type noise : gaussian
  # s&p
  # speckle
  # trian - gauss_img = torch.tensor(random_noise(imgs, mode='gaussian', mean=0, var=0.05, clip=True))
  gauss_img = torch.tensor(random_noise(imgs, mode=noise_type, mean=mean, var=var, clip=True))
  return gauss_img

def train(train_dataloader):

    # Init model
    autoencoder = Autoencoder().to(device)
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    # set to training mode
    autoencoder.train()

    train_loss_avg = []

    print('Training ...')
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0
        
        for image_batch, _ in train_dataloader:
            
            image_batch = image_batch
            
            # autoencoder reconstruction
            noise_img = add_noise(image_batch, 'gaussian')
            image_batch_recon = autoencoder(noise_img.float().to(device))
            
            # reconstruction error
            loss = F.mse_loss(image_batch_recon, image_batch.to(device))
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
        
    # Plot train
    fig = plt.figure()
    plt.plot(train_loss_avg)
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction error')
    plt.show()

    return autoencoder

def test (autoencoder, test_dataloader):

    # set to evaluation mode
    autoencoder.eval()

    test_loss_avg, num_batches = 0, 0
    for image_batch, _ in test_dataloader:
        
        with torch.no_grad():

            image_batch = image_batch
            noise_img = add_noise(image_batch, 'gaussian')
            # autoencoder reconstruction
            image_batch_recon = autoencoder(noise_img.float())

            # reconstruction error
            loss = F.mse_loss(image_batch_recon, image_batch)

            test_loss_avg += loss.item()
            num_batches += 1
        
    test_loss_avg /= num_batches
    print('average reconstruction error: %f' % (test_loss_avg))


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model, noise_type, mean=0, var=0.05):

    with torch.no_grad():

        images = images
        # add noise
        noise_img = add_noise(images, noise_type, mean, var)
        images = model(noise_img.float())
        images = images.cpu()
        images = to_img(images)


        noise_img = noise_img.cpu()
        noise_img = to_img(noise_img)

        # show image noise
        print('Input image: ')
        np_imagegrid = torchvision.utils.make_grid(noise_img[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

        # Show output 
        print('Denoising Autoencoder reconstruction:')
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()


def main():
    # Load dataset
    train_dataloader, test_dataloader = load_data()
    
    # Train model
    autoencoder = train(train_dataloader)

    # test 
    test(autoencoder, test_dataloader)


    # Test model
    images, labels = iter(test_dataloader).next()

    # First visualise the original images
    print('Original images')
    show_image(torchvision.utils.make_grid(images[1:50],10,5))
    plt.show()

    # Reconstruct and visualise the images using the autoencoder
    visualise_output(images, autoencoder, 'gaussian')
    visualise_output(images, autoencoder, 'gaussian', 0 , 0.25)


if __name__ == "__main__":
    main()