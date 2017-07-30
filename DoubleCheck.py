import time
import base64
import numpy as np
import cv2
from test_ssd_512 import detect_img
import json
import utils
# from Main import Main

class DoubleCheck:
    def __init__(self, context, load_queue_length=1):
        self.load_queue_length = load_queue_length
        self.context = context
        self.is_detected = 0
        pass


    def tclass_edge_detector(self, img, tclass):
        """
        detect goods whether touch border
        """
        tclasses = []
        tscores = []
        tbboxes = []

        height = np.shape(img)[0]
        width = np.shape(img)[1]
        border_ymax = height * self.context.edge_detector.prop_border
        rclasses, rscores, rbboxes = detect_img(img)

        utils.plt_bboxes_cv(img, rclasses, rscores, rbboxes)

        for i in range(rclasses.shape[0]):
            ymin = float(rbboxes[i, 0] * height)
            xmin = float(rbboxes[i, 1] * width)
            ymax = float(rbboxes[i, 2] * height)
            xmax = float(rbboxes[i, 3] * width)

            if ymin < border_ymax and (rclasses[i] == tclass): # only for oreo and TesDi
                tclasses.append(rclasses[i])
                tscores.append(rscores[i])
                tbboxes.append(rbboxes[i])
                print("border detect", i)

        return  tclasses, tscores, tbboxes

    def conduct(self):
        k = self.context.redis_connection_img_queue.keys()
        k.sort()
        if len(k) is 0:
            print("EdgeDetection: Queue is empty")
            return

        for img_b64 in self.context.redis_connection_img_queue.mget(k[-self.load_queue_length:]):
            t = time.time()
            img_jpg = base64.b64decode(img_b64)
            # data = np.asarray(bytearray(img_jpg), dtype=np.uint8).reshape(512,512,3)[:,:,::-1]
            data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
            data = cv2.imdecode(data, 1)
            now_tclasses, now_tscores, now_tbboxes = self.tclass_edge_detector(data, self.context.behaviour_detector._tclass)
            if len(now_tclasses) is not 0:
                self.is_detected = 1
            else:
                self.is_detected = 2