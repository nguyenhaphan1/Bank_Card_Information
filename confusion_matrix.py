import pandas as pd
import numpy as np
import json

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
pred = pd.read_csv('output_tutorial/pred.csv')
truth = pd.read_csv('output_tutorial/train/annotations.csv')

with open('label_map.pbtxt', 'r') as f:
    category = f.read().replace("name:", '"name":').replace("id", '"id"').replace('\n', "").replace("item", "")
    f.close()
# print(category)
categories = json.loads(category)
print(type(categories[0]))
print(len(categories))
output_path = 'output_tutorial/confusion_matrix.csv'
def compute_iou(groundtruth_box, detection_box):
    g_xmin, g_ymin, g_xmax, g_ymax = tuple(groundtruth_box)
    d_xmin, d_ymin, d_xmax, d_ymax = tuple(detection_box)

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def process_detections(test, categories, pred):
    confusion_matrix = np.zeros(shape=(len(categories) +1, len(categories) + 1))
    file_unique = test['filename'].unique()
    for file in file_unique:
        print(file)
        test_df = test[test['filename']==file]
        test_df.reset_index(inplace = True, drop = True)
        pred_df = pred[pred['filename']==file]
        pred_df.reset_index(inplace = True, drop = True)
        pred_class = pred_df[pred_df['score']>= CONFIDENCE_THRESHOLD]
        groundtruth_boxes = test_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        print(groundtruth_boxes)
        detection_boxes = pred_class[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        print(detection_boxes)
        matches = []
        for i in range(len(groundtruth_boxes)):
            for j in range(len(detection_boxes)):
                iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])
                if iou > IOU_THRESHOLD:
                    matches.append([i, j, iou])
        matches = np.array(matches)

        if matches.shape[0] > 0:
            #Sắp xếp list các bbox trùng nhau theo thứ tự giảm dần giá trị IOU để
            #loại bỏ những giá trị trùng và giữ lại box có IOU cao nhất
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
            #Bỏ duplicate
            matches = matches[np.unique(matches[:,1], return_index = True)[1]]

            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            # print(matches)
        for i in range(len(groundtruth_boxes)):
            print(i)
            if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                print("inside :", i)

                confusion_matrix[categories[test_df['class'][i]]['id'] - 1][categories[pred_class['class'][matches[matches[:, 0] == i].tolist()[0][1]]]['id'] - 1] += 1
            else:
                confusion_matrix[categories[test_df['class'][i]]['id'] - 1][confusion_matrix.shape[1] - 1] += 1

        for i in range(len(detection_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
                confusion_matrix[confusion_matrix.shape[0] - 1][categories[pred_class['class'][i]]['id'] - 1] += 1
    return confusion_matrix


def display(confusion_matrix, test, categories, output_path):
    results = []
    class_uniq = test['class'].unique()
    for label in class_uniq:
        class_id = int(float(categories[label]['id'])) - 1
        name = label

        total_target = np.sum(confusion_matrix[class_id, :])
        total_predicted = np.sum(confusion_matrix[:, class_id])

        precision = float(confusion_matrix[class_id, class_id] / total_predicted)
        recall = float(confusion_matrix[class_id, class_id] / total_target)

        results.append({'category': name, 'precision_@{}IOU'.format(IOU_THRESHOLD): precision,
                        'recall_@{}IOU'.format(IOU_THRESHOLD): recall})
    print(confusion_matrix)
    print(precision)
    print(recall)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_path)

if __name__ == '__main__':
    confusion_matrix = process_detections(truth, categories, pred)
    display(confusion_matrix, truth, categories, output_path)
