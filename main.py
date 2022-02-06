import ModelTrainer
import SVMTrainer
import Utils
import VisualizationUtils

#Init
Utils.set_classes()

## Trenowanie
train_data_loader, validation_data_loader, test_data_loader = Utils.prepare_data()
model_trainer = ModelTrainer.ModelTrainer()
model = model_trainer.train_resnet(train_data_loader, validation_data_loader, num_epochs=25)
model = torch.load('models/resnet_fc.pth', map_location=torch.device('cpu'))model.eval()
Utils.predict_image(model, test_data_loader)

# model_trainer = SVMTrainer.SVMTrainer()
# model, svm = model_trainer.train_svm(train_data_loader, validation_data_loader, type='rbf', degree=2)
# model.eval()
# Utils.predict_image_svm(model, svm, test_data_loader)

## 3a
data_loader = Utils.get_all_data()
model = torch.load('models/resnet_fc.pth', map_location=torch.device('cpu'))
model.eval()
wrong_predicted_ids = VisualizationUtils.show_wrong_predictions(model, data_loader)
Utils.save_wrong_predicted(wrong_predicted_ids)