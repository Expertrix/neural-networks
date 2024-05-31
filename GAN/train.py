import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from model import D, G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + device.type)

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST("dataset/", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

epochs = 10
lr = 2e-4
loss = nn.BCELoss()
dis = D().to(device)
gen = G().to(device)
genOptim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
disOptim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
for epoch in range(epochs):
    for iteration, (imgs, _) in enumerate(train_loader):
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
        disOptim.zero_grad()
        d_loss.backward()
        disOptim.step()

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)

        fake_inputs = gen(noise)
        fake_outputs = dis(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
        g_loss = loss(fake_outputs, fake_targets)
        genOptim.zero_grad()
        g_loss.backward()
        genOptim.step()

        if iteration % 100 == 0 or iteration == len(train_loader):
            print("Epoch {} Iteration {}: d_loss {:.3f} g_loss {:.3f}".format(epoch, iteration, d_loss.item(), g_loss.item()))

    if (epoch + 1) % 10 == 0: torch.save(dis, "e{}.pth".format(epoch + 1))