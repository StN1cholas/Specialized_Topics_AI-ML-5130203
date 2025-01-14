import torch.nn as nn
from torchvision import models as tmodels
from django.db import models
import json

classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
           'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
           'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
           'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear',
           'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
           'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
           'turnip', 'watermelon']


class ImagePrediction(models.Model):
    image = models.ImageField(
        upload_to='images/')  # Загрузка изображений в папку 'images/'
    predictions = models.TextField()  # JSON-строка с предсказаниями
    uploaded_at = models.DateTimeField(
        auto_now_add=True)  # Дата и время загрузки

    def save(self, *args, **kwargs):
        if self.predictions:
            predictions_data = json.loads(self.predictions)

            # Формируем словарь {вероятность: класс}
            predictions_dict = {predictions_data[i]: classes[i] for i in
                                range(len(classes))}

            # Сортируем словарь по убыванию вероятностей
            sorted_predictions = dict(
                sorted(predictions_dict.items(), key=lambda item: item[0],
                       reverse=True))

            # Переводим отсортированный словарь в строку для сохранения
            self.predictions = json.dumps(sorted_predictions)

        # Сохраняем объект
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Prediction uploaded at {self.uploaded_at}"


class ResNet50Custom(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Custom, self).__init__()

        self.backbone = tmodels.resnet50(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
