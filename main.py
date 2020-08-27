import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import urllib
import os
import csv
import cv2
import time
from PIL import Image
import glob
import xml.etree.ElementTree as ET
# from google.colab import files

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# register_matplotlib_converters()
# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            curr_path = root.find('path').text
            curr_path.replace("\\","/")
            value = (curr_path,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df_pre = pd.DataFrame(xml_list, columns=column_name)
    xml_df = xml_df_pre.drop(['width', 'height'], axis=1)
    cols = xml_df.columns.tolist()
    cols = [cols[0], cols[2], cols[3], cols[4], cols[5], cols[1]]
    xml_df = xml_df[cols]
    return xml_df


image_path = 'C:/Users/Ruben/Documents/OSRS Botting/osrs_botting/objectdetection/models/annotations'
xml_df = xml_to_csv(image_path)

def show_image_objects(image_row):
    img_path = image_row.filename
    box = [
    image_row.xmin, image_row.ymin, image_row.xmax, image_row.ymax
    ]

    image = read_image_bgr(img_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_box(draw, box, color=(255, 255, 0))

    plt.axis('off')
    plt.imshow(draw)
    plt.show()

# show_image_objects(xml_df.iloc[0])

## PreProcessing
train_df, test_df = train_test_split(xml_df, test_size=0.2, random_state=RANDOM_SEED)

ANNOTATIONS_FILE = 'annotations.csv'
CLASSES_FILE = 'classes.csv'

train_df.to_csv(ANNOTATIONS_FILE, index=False, header=None)
classes = set(['tree'])
with open(CLASSES_FILE, 'w') as f:
  for i, line in enumerate(sorted(classes)):
    f.write('{},{}\n'.format(line,i))

PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'

# URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
# urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

# print('Downloaded pretrained model to ' + PRETRAINED_MODEL)

# python keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights snapshots/_pretrained_model.h5 --batch-size 8 --steps 500 --epochs 10 csv annotations.csv classes.csv


model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
print(model_path)

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

labels_to_names = pd.read_csv(CLASSES_FILE, header=None).T.loc[0].to_dict()

def predict(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
    )

    boxes /= scale

    return boxes, scores, labels

THRES_SCORE = 0.6

def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(image, b, caption)

def show_detected_objects(image_row):
    img_path = image_row.filename

    image = read_image_bgr(img_path)

    boxes, scores, labels = predict(image)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # true_box = [
    # image_row.xmin, image_row.ymin, image_row.xmax, image_row.ymax
    # ]
    # draw_box(draw, true_box, color=(255, 255, 0))

    draw_detections(draw, boxes, scores, labels)

    plt.axis('off')
    plt.imshow(draw)
    plt.show()

running = True
while running:
    input_int = input("Choose an image (INTEGER) to view or type q to exit: ")
    if input_int == 'q':
        break
    show_detected_objects(test_df.iloc[int(input_int)])
