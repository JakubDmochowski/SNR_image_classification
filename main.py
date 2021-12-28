import torch

import ModelTrainer
import Utils

Utils.set_classes()
train_data_loader, validation_data_loader, test_data_loader = Utils.prepare_data()

model_trainer = ModelTrainer.ModelTrainer()
model = model_trainer.train_resnet(train_data_loader, validation_data_loader, num_epochs=25)
# model = torch.load('resnet1.pth')
model.eval()
Utils.predict_image(model, test_data_loader)

