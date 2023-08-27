import torch
from PGD import LinfPGD
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from model.model import * 
import torch.optim as optim
from tqdm import tqdm



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
train_data = datasets.CIFAR10(root='./dataset/',train=True,download = False,transform=transform_train)
test_data = datasets.CIFAR10(root='./dataset/',train=False,download= False,transform=transform_test)

train_loader = DataLoader(dataset=train_data,batch_size=128,shuffle=True,num_workers=4)
test_loader = DataLoader(dataset=test_data,batch_size=128,shuffle=False,num_workers=4)

criterion = nn.CrossEntropyLoss()
model = ResNet18()
optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0002)
adversary = LinfPGD(model)
def train_PGD():
    model.train()
    epochs = 2
    for epoch in range(epochs):
        current_loss = 0
        for step,(batch_x,batch_y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            adv_x = adversary.perturb(batch_x,batch_y)
            output = model(adv_x)
            loss = criterion(output,batch_y)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            print(f"Epoch:{epoch}  Step:{step}   Loss:{current_loss}")


def test_PGD():
    model.eval()
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for step,(batch_x,batch_y) in enumerate(tqdm(test_loader)):
            total += batch_y.size(0)
            output = model(batch_x)
            indexes = torch.argmax(output,dim=1)
            benign_correct += (indexes == batch_y).sum().item()


            adv_image = adversary.perturb(batch_x,batch_y)
            outputs = model(adv_image)
            indexes = torch.argmax(outputs,dim = 1)
            adv_correct += (indexes == batch_y).sum().item()
    print(f"benign accuracy : {100 * benign_correct / total}%")
    print(f"adversary accuracy : {100 * adv_correct / total}%")
    





def main():
    train_PGD()
    test_PGD()
if __name__ == "__main__":
    main()


