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

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    CNN = Net()
    CNN = CNN.to(device)

    PATH = '/saved_model/save1.pt'
    checkpoint = torch.load(PATH)

    CNN.load_state_dict(checkpoint['model_state_dict'])

    test_trainsform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    path_val_test = "./test"

    val_test = datasets.ImageFolder(path_val_test, transform = test_trainsform)
    val_testing_loader = data.DataLoader(
        val_test,
        shuffle=True,
        batch_size = 1,
        num_workers= 4
    )

    loss_function = nn.CrossEntropyLoss()

    correct = 0
    for batch_size, (x, y) in enumerate(val_testing_loader):
        CNN.eval()

        x = x.to(device)
        y = y.to(device)

        output = CNN(x)

        loss = loss_function(output, y).item()
        pred = output.argmax()

        correct += pred.eq(y.item())

    print("{}/{}".format(correct, len(val_testing_loader.dataset)))

if __name__ == "__main__":
    test()