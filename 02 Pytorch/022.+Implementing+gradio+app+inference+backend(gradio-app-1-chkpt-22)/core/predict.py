import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

class CustomCnnModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomCnnModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Compute dynamically the input size for fully connected layers
        self._to_linear = None
        self._get_conv_output(self.input_dim)


        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def _get_conv_output(self, input_dim=224):
      with torch.no_grad():
          dummy_input = torch.zeros(1, 3, input_dim, input_dim)
          output = self.conv_layers(dummy_input)
          self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
      x = self.conv_layers(x)
      x = x.view(x.size(0), -1)  # Flatten
      x = self.fc_layers(x)
      return x
    
class ImageClassifier():
    def __init__(self, model_path, class_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomCnnModel(input_dim=128, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if class_name is None:
            self.class_name = {0: 'Cat', 1: 'Dog', 2: 'person'}
        else:
            self.class_name = class_name
        

        self.transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )


    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            # [[0.3, 0.7, 0.9]]
            _, predicted = torch.max(output, 1)

        label = self.class_name[predicted.item()]

        img = cv2.imread(image_path)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        output_path = "labeled_image.jpg"
        cv2.imwrite(output_path, img)
        cwd = os.getcwd()
        output_path = os.path.join(cwd, output_path)

        return label, output_path
        