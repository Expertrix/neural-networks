import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import model

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST("dataset/", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dis = torch.load("e10.pth", map_location=device)
dis.eval()  

gen = model.G().to(torch.device('cuda'))
gen.eval()

loss = nn.BCELoss()
for iteration, (imgs, _) in enumerate(train_loader):
    if(iteration == 10): break
    iteration += 1

    real_inputs = imgs.to(device)
    real_outputs = dis(real_inputs)
    real_label = torch.ones(real_inputs.shape[0], 1).to(device)

    noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
    noise = noise.to(device)
    fake_inputs = gen(noise)
    fake_outputs = dis(fake_inputs)
    fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

    outputs = torch.cat((real_outputs, fake_outputs), 0)
    targets = torch.cat((real_label, fake_label), 0)

    d_loss = loss(outputs, targets)

    print("Iteration {}: d_loss {:.3f}".format(iteration, d_loss.item()))