import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import torch.utils.data as data
from pathlib import Path
import torchvision.models as models


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.model = models.vgg19_bn(pretrained = True)
    self.model.classifier._modules['6'] = nn.Linear(4096, 2)

  def forward(self, x):
    x = self.model(x)

    return x

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    number_work_ = 4
    batch_size_ = 32
    LR_ = 0.000001

    path_val_train = '/data/train'

    train_trainsform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    val_train = datasets.ImageFolder(path_val_train, transform = train_trainsform)

    val_training_loader = data.DataLoader(val_train, batch_size = batch_size_, shuffle = True, num_workers = number_work_)

    CNN = Net()
    CNN = CNN.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,CNN.parameters()), lr = LR_)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_function = nn.CrossEntropyLoss()

    def val(data):
        CNN.eval()
        correct = 0
        with torch.no_grad():
            for batch_size, (x, y) in enumerate(data):
                x = x.to(device)
                y = y.to(device)
                output = CNN(x)
                test_loss = loss_function(output, y).item()
                pred = output.argmax(dim = 1, keepdim = True)
                correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= len(data.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data.dataset),
            00. * correct / len(data.dataset)))
        CNN.train()


    CNN.train()
    for epoch in range(0,30):
        train_loss = 0
        for batch_size_, (x, y) in enumerate(val_training_loader):
            x = x.to(device)
            y = y.to(device)
            output = CNN(x)
            optimizer.zero_grad()
            loss = loss_function(output, y)
            loss.backward()       #autograd
            optimizer.step()      #update
            scheduler.step()
            train_loss += loss.item()
        print("training : ", train_loss / len(val_training_loader))
        val(val_training_loader)
        torch.save({
            'epoch':epoch,
            'model_state_dict':CNN.state_dict(),
            'optimizer':optimizer.state_dict(),
            'loss':loss},
            "/saved_model/save1.pt")

if __name__ == "__main__":
    train()
