import torch
import torch.nn as nn

from PIL import Image
from os import listdir

import torchvision.transforms.functional as fct

image_names = sorted(listdir('/home/eduard/Private/sgan-modified/customdata/images'))
print(image_names)

img_array = []

for img_name in image_names:
    full_image_path = "/home/eduard/Private/sgan-modified/customdata/images/{}".format(img_name)
    img = Image.open(full_image_path)
    img = fct.to_tensor(img)
    img = img[:3, :, :]
    img_array.append(img)


def retrieve_image_from_coordinate(x, y):
    batch_num = -1
    subset_num = -1

    if 150 <= y < 260:
        subset_num = 0
    elif 50 <= y < 150:
        subset_num = 1
    elif -50 <= y < 50:
        subset_num = 2
    else:
        subset_num = 3

    if -250 <= x < -125:
        batch_num = 0
    elif -125 <= x < -50:
        batch_num = 1
    elif -50 <= x < 50:
        batch_num = 2
    elif 50 <= x < 100:
        batch_num = 3
    elif 100 <= x <= 150:
        batch_num = 4
    elif 150 <= x < 250:
        batch_num = 5

    img_num = 4 * batch_num + subset_num

    return img_array[img_num]


temp_img = retrieve_image_from_coordinate(-72, -150)
class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 7)
        self.conv2 = nn.Conv2d(10, 16, 7)
        self.conv3 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(8 * 71 * 96, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 32)

        self.pool = nn.MaxPool2d(4, 4)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, image):
        image = image.unsqueeze(dim=0)
        print(image.shape)
        x = self.conv1(image)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

# im = fct.to_pil_image(temp_img)
# im.show()




enc = ImageEncoder()
enc(temp_img)
