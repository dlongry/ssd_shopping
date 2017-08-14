import time
import base64
import numpy as np
import cv2
from test_ssd_512 import detect_img
import json
import utils


class EdgeDetection:
    def __init__(self, context,load_queue_length=1, prop_border=0.1):
        self.load_queue_length = load_queue_length
        self.prop_border = prop_border
        self.context = context
        self.is_detected = False
        self.now_tclasses = None
        self.now_tscores = None
        self.now_tbboxesr = None
        self.ori_places = None
        self.last_frame = None
        pass


    def edge_detector(self, img):
        """
        detect goods whether touch border
        """
        tclasses = []
        tscores = []
        tbboxes = []
        ori_places=[]

        height = np.shape(img)[0]
        width = np.shape(img)[1]
        border_ymax = height * self.prop_border
        rclasses, rscores, rbboxes = detect_img(img)

        utils.plt_bboxes_cv(img, rclasses, rscores, rbboxes)

        #save n sec detection results to redis
        # d = {
        #     'rclasses': rclasses.tolist(),
        #     'rscores': rscores.tolist(),
        #     'rbboxes': rbboxes.tolist()
        # }
        # self.context.redis_connection_intermediate_queue.set(int(time.time()*100000), json.dumps(d), px=300)

        for i in range(rclasses.shape[0]):
            ymin = float(rbboxes[i, 0] * height)
            xmin = float(rbboxes[i, 1] * width)
            ymax = float(rbboxes[i, 2] * height)
            xmax = float(rbboxes[i, 3] * width)

            # if  20<ymin < border_ymax and (rclasses[i] == 1 or rclasses[i] == 2): # only for oreo and TesDi
            if  ymin < border_ymax : # only for oreo and TesDi
                tclasses.append(rclasses[i])
                tscores.append(rscores[i])
                tbboxes.append(rbboxes[i])
                # print("border detect", i)

        return  tclasses, tscores, tbboxes

    def conduct(self):
        k = self.context.redis_connection_img_queue.keys()
        k.sort()
        if len(k) is 0:
            # print("EdgeDetection: Queue is empty")
            return

        for img_b64 in self.context.redis_connection_img_queue.mget(k[-self.load_queue_length:]):
            t = time.time()
            img_jpg = base64.b64decode(img_b64)
            # data = np.asarray(bytearray(img_jpg), dtype=np.uint8).reshape(512,512,3)[:,:,::-1]
            data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
            data = cv2.imdecode(data, 1)
            data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
            now_tclasses, now_tscores, now_tbboxes = self.edge_detector(data)
            if len(now_tclasses) is not 0:
                self.now_tclasses, self.now_tscores, self.now_tbboxesr =  now_tclasses, now_tscores, now_tbboxes
                if  self.now_tscores > 0.88:
                    self.last_frame = data.copy()
                    self.is_detected = True
