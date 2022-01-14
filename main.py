import ModelTrainer
import SVMTrainer
import Utils

Utils.set_classes()
train_data_loader, validation_data_loader, test_data_loader = Utils.prepare_data()

model_trainer = ModelTrainer.ModelTrainer()
model = model_trainer.train_resnet(train_data_loader, validation_data_loader, num_epochs=25)
model.eval()
Utils.predict_image(model, test_data_loader)

# model_trainer = SVMTrainer.SVMTrainer()
# model, svm = model_trainer.train_svm(train_data_loader, validation_data_loader, type='rbf', degree=2)
# model.eval()
# Utils.predict_image_svm(model, svm, test_data_loader)
