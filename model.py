import os #import OS in order to make GPU visible
import tensorflow as tf
import sys
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
import detection
import cv2


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path2scripts = 'models//research' # provide pass to the research folder
sys.path.insert(0, path2scripts) # make scripts in models/research available for import

#Path to pipeline.config file of the model
path2config ='C:/Users/dumbw/Documents/pythonProject/BankCardExtraction/workspace/exported_models/ssd_resnet50_v3/pipeline.config'
#Path to folder saving model train checkpoint
path2model = 'C:/Users/dumbw/Documents/pythonProject/BankCardExtraction/workspace/exported_models/ssd_resnet50_v3/checkpoint'

configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()

path2label_map = 'workspace/data/train/label_map.pbtxt' # TODO: provide a path to the label map file
category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)

def detect_img(img):
    return detection.detect_image(img, category_index=category_index, detection_model=detection_model, box_th=0.3)

# image = cv2.imread('C:/Users/dumbw/OneDrive/Desktop/VNPT_INTERN/Project Extract Number Credit Card/data_raw/hard/a.jpg')
# image = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA)
# print(detect_img(image))