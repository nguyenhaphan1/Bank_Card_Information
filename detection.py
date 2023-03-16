import os #import OS in order to make GPU visible
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from object_detection.utils import visualization_utils as viz_utils


def detect_fn(image, detection_model):
    """
    Detect objects in image.

    Args:
      image: (tf.tensor): 4D input image

    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """
    # return cv2.imread(path)
    return cv2.resize(cv2.imread(path), (640, 640), interpolation = cv2.INTER_AREA)

def inference_with_plot(path2images, box_th=0.5, PATH = None, category_index=None):
    if PATH == None:
        print('No path has providing, please provide a valid path to image')
        return
    if category_index == None:
        print('No category found is empty, please provide categories')
    count = 0
    """
    Function that performs inference and plots resulting b-boxes

    Args:
      path2images: an array with pathes to images
      box_th: (float) value that defines threshold for model prediction.

    Returns:
      None
    """
    for image_path in path2images:

        print('Running inference for {}... '.format(image_path), end='')
        try:
            image_np = load_image_into_numpy_array(image_path)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}

            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=16,
                min_score_thresh=box_th,
                agnostic_mode=False,
                line_thickness=2, skip_scores=True
            )

            plt.figure(figsize=(15, 10))
            #         plt.imshow(image_np_with_detections)
            numb_str = extract_card_number(
                detection_boxes=detections['detection_boxes'],
                detection_classes=detections['detection_classes'],
                detection_scores=detections['detection_scores'],
                th=box_th)

            drawed_image = draw_number_on_image(numb_str, image_np_with_detections)
            cv2.imwrite(os.path.join(PATH, str(count) + '.jpg'), drawed_image)
            count += 1
            print('Done')
        except Exception as exception:
            print(exception)
            pass
#     plt.show()

def detect_image(image_np, box_th=0.5, category_index = {}, detection_model = None):
    try:
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}

        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=16,
                min_score_thresh=box_th,
                agnostic_mode=False,
                line_thickness=2,skip_scores= False)

        numb_str = extract_card_number(
            detection_boxes=detections['detection_boxes'],
            detection_classes=detections['detection_classes'],
            detection_scores=detections['detection_scores'],
            th=box_th)

        drawed_image = draw_number_on_image(numb_str, image_np_with_detections)
    except Exception as exception:
        print(exception)
        return None
    return numb_str, drawed_image

def nms(rects, thd=0.5):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []

    remove = [False] * len(rects)

    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


def inference_as_raw_output(path2images,
                            box_th=0.25,
                            nms_th=0.5,
                            to_file=False,
                            data=None,
                            path2dir=False):
    """
   Function that performs inference and return filtered predictions

   Args:
     path2images: an array with pathes to images
     box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
     nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
     to_file: (boolean). When passed as True => results are saved into a file. Writing format is
     path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
     data: (str) name of the dataset you passed in (e.g. test/validation)
     path2dir: (str). Should be passed if path2images has only basenames. If full pathes provided => set False.

   Returs:
     detections (dict): filtered predictions that model made
   """
    print(f'Current data set is {data}')
    print(f'Ready to start inference on {len(path2images)} images!')

    for image_path in tqdm(path2images):

        if path2dir:  # if a path to a directory where images are stored was passed in
            image_path = os.path.join(path2dir, image_path.strip())

        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # checking how many detections we got
        num_detections = int(detections.pop('num_detections'))

        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # defining what we need from the resulting detection dict that we got from model output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

        # filtering out detection dict in order to get only boxes, classes and scores
        detections = {key: value for key, value in detections.items() if key in key_of_interest}

        if box_th:  # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array

        if nms_th:  # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                   detections['detection_scores'],
                                   detections['detection_classes']
                                   )
                               )
            boxes, scores, classes = nms(output_info)

            detections['detection_boxes'] = boxes  # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes

        if to_file and data:  # if saving to txt file was requested

            image_h, image_w, _ = image_np.shape
            file_name = f'pred_result_{data}.txt'

            line2write = list()
            line2write.append(os.path.basename(image_path))

            with open(file_name, 'a+') as text_file:
                # iterating over boxes
                for b, s, c in zip(boxes, scores, classes):
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w

                    list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                    line2append = ','.join([str(item) for item in list2append])

                    line2write.append(line2append)

                line2write = ' '.join(line2write)
                text_file.write(line2write + os.linesep)

        return detections


def extract_card_number(detection_scores, detection_classes, detection_boxes, th=0.5):
    if len(detection_scores) < 16:
        print("Can't read 16 digits of this card. Please try a clearer image.")
        return
    number_str = ''
    score_correspond_boxes = [x for _, x in sorted(zip(detection_boxes[:16, 1], detection_scores[:16]))]
    class_correspond_boxes = [x for _, x in sorted(zip(detection_boxes[:16, 1], detection_classes[:16]))]
    count = 0
    for i in range(16):
        if score_correspond_boxes[i] >= th:
            number_str += str(int(class_correspond_boxes[i]))
            count += 1
        if count == 4:
            count = 0
            number_str += ' '

    return number_str.strip()

def draw_number_on_image(numb_str, image, org = (50, 50), font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 0), thickness = 1):
    return cv2.putText(image, numb_str, org, font, fontScale, color, thickness, cv2.LINE_AA)
