from __future__ import print_function, division

import copy
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import svm
from torch.utils.data import DataLoader
from torchvision import models
import pickle

import Utils


class SVMTrainer:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_accuracy = list()
        self.validation_accuracy = list()
        self.model = None
        self.svm = None

    def train_svm(self, train_data_loader, test_data_loader, type='linear', degree=2):
        model = models.resnet18(pretrained=True)
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        # Override classifier part
        model.fc = nn.ReLU()
        model = model.to(self.device)

        SVM = svm.SVC(kernel=type, degree=degree)
        # Clear lists before training
        self.training_accuracy = list()
        self.validation_accuracy = list()

        self.model, self.svm = self.train_svm_model(train_data_loader, test_data_loader, model, SVM)

        # Utils.visualize_model(test_data_loader, model, device)
        torch.save(model, f'resnet-svm-{type}.pth')

        pickle.dump(SVM, open(f'models/resnet-svm-{type}.sav', 'wb'))

        plt.plot(self.training_accuracy, label='Training accuracy')
        plt.plot(self.validation_accuracy, label='Validation accuracy')
        plt.legend(frameon=False)
        plt.show()
        plt.ioff()
        plt.show()

        plt.savefig(f'plots/svm-{type}-acc.pdf')

        return self.model, self.svm

    def train_svm_model(self, train_data_loader: DataLoader, test_data_loader: DataLoader,
                    model, SVM: svm.OneClassSVM):
        since = time.time()

        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_data_loader
            else:
                model.eval()  # Set model to evaluate mode
                data_loader = test_data_loader

            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model_outputs = model(inputs)
                    model_outputs = model_outputs.cpu()
                    labels = labels.to("cpu")
                    if phase == 'train':
                        SVM.fit(model_outputs, labels)
                    preds = SVM.predict(model_outputs)

                # statistics
                preds = torch.from_numpy(preds)
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            # Save statistics
            if phase == 'train':
                self.training_accuracy.append(epoch_acc)
            else:
                self.validation_accuracy.append(epoch_acc)

            print('{} Acc: {:.4f}'.format(phase, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return model, SVM
