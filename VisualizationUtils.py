import matplotlib.pyplot as plt
import numpy as np
import torch
import Utils


def show_wrong_predictions(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig = plt.figure()
    iteration = 0
    wrong_predict_id = []
    for inputs, labels in data_loader:
        pred = predict_image(model, inputs)
        if pred != labels:
            wrong_predict_id.append(iteration)
            predicted_class = Utils.classes[pred[0].numpy().flat[0]]
            original_class = Utils.classes[labels[0].numpy().flat[0]]
            filename = "images/" + str(iteration) + '.png'
            imshow(inputs[0], filename,  title='original: {}, predicted: {}'.format(original_class, predicted_class))
        iteration += 1

    return wrong_predict_id

def predict_image(model, image):
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return preds

def visualize_model(test_data_loader, model, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def imshow(inp, filename, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    fig = plt.figure()
    ax = fig.subplots()
    ax.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig(filename)
    plt.show()

