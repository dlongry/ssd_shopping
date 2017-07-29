import cv2
import random


def plt_bboxes_cv(img, classes, scores, bboxes, linewidth=3, type=False):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if score < 0.5:
                continue
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())

            if type:
                ymin = int(bboxes[i, 0])
                xmin = int(bboxes[i, 1])
                ymax = int(bboxes[i, 2])
                xmax = int(bboxes[i, 3])
            else:
                ymin = int(bboxes[i, 0] * height)
                xmin = int(bboxes[i, 1] * width)
                ymax = int(bboxes[i, 2] * height)
                xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cls_id], linewidth)
            class_name = str(cls_id)
            # font=cv.InitFont(cv2.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
            # cv2.cv.PutText(img, '{:s} | {:.3f}'.format(class_name, score), (xmin, ymin - 2), font, (0,255,0))

            font = cv2.FONT_HERSHEY_TRIPLEX
            # cv2.putTe
            cv2.putText(img, '{:s} | {:.3f}'.format(class_name, score), (xmin, ymin + 15), font, 0.8, (0, 255, 0), 1,
                        False)
    cv2.imshow('Object detecting...', img)
    cv2.waitKey(1)


def plt_tracker(img, classes, bboxes, linewidth=3):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    height = img.shape[0]
    width = img.shape[1]
    cls_id = int(classes)
    color = (random.random(), random.random(), random.random())

    ymin = int(bboxes[1])
    xmin = int(bboxes[0])
    h = int(bboxes[3])
    w = int(bboxes[2])

    cv2.rectangle(img, (xmin, ymin), (xmin+w, ymin+h), color, linewidth)
    cv2.imshow('Object detecting...', img)
    cv2.waitKey(1)
