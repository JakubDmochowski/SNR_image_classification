import torch
import torchvision.models as models
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_alex(train_loader, test_loader):
    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    print("Using", device)

    model = models.resnet18(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                     nn.Linear(in_features=9216, out_features=4096, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5, inplace=False),
                                     nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(in_features=4096, out_features=1000, bias=True),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(test_loader))
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy / len(test_loader):.3f}")
                running_loss = 0
                model.train()

    torch.save(model, 'alexnet.pth')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


def test_model(valid_loader):
    model = torch.load('alexnet.pth')
    model.eval()
    to_pil = transforms.ToPILImage()
    fig = plt.figure(figsize=(10, 10))
    for images, labels in valid_loader:
        for ii in range(len(images)):
            image = to_pil(images[ii])
            index = predict_image(image, model)
            sub = fig.add_subplot(1, len(images), ii + 1)
            res = int(labels[ii]) == index
            # sub.set_title(str(classes[index]) + ":" + str(res))
            plt.axis('off')
            plt.imshow(image)
        plt.show()


def predict_image(image, model):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index
