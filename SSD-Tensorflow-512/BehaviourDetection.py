from test_ssd_512 import detect_img
import numpy as np
import cv2
import base64
import time
import json
import KCF
import utils
from test_ssd_512 import detect_img


class BehaviourDetection:
    def __init__(self, context, load_queue_length=1, img_w=512, img_h=512):
        self.load_queue_length = load_queue_length
        self.context = context
        self.img_w = img_w
        self.img_h = img_h
        self.is_detected = 0
        self._tclass = self.context.edge_detector.now_tclasses[0]
        self.tracker = KCF.kcftracker(True, True, False,
                                      True)  # hog, fixed_window, multiscale, lab(True, True, False,lab)
        self.ix = int(self.context.edge_detector.now_tbboxesr[0][1] * img_w)
        self.iy = int(self.context.edge_detector.now_tbboxesr[0][0] * img_h)  # init (x0,y0)
        self.w = int(self.context.edge_detector.now_tbboxesr[0][3] * img_w) - int(
            self.context.edge_detector.now_tbboxesr[0][1] * img_w)
        self.h = int(self.context.edge_detector.now_tbboxesr[0][2] * img_h) - int(
            self.context.edge_detector.now_tbboxesr[0][0] * img_h)

        self.ori_y = int(self.context.edge_detector.now_tbboxesr[0][0] * img_h)

        self.tracker.init([int(self.ix+0.1*self.w), int(self.iy+0.1*self.h), int(self.w*0.8), int(self.h*0.8)], self.context.edge_detector.last_frame)

        self.dis_threshold = self.context.edge_detector.prop_border * self.img_h
        self.move_dis = 0

        print("into behaviour state,tclass:", self._tclass)

        pass
    def tclass_edge_detector(self, img,tclass):
        """
        detect goods whether touch border
        """
        tclasses = []
        tscores = []
        tbboxes = []
        ori_places=[]

        height = np.shape(img)[0]
        width = np.shape(img)[1]
        border_ymax = height * self.context.edge_detector.prop_border
        rclasses, rscores, rbboxes = detect_img(img)


        for i in range(rclasses.shape[0]):
            ymin = float(rbboxes[i, 0] * height)
            xmin = float(rbboxes[i, 1] * width)
            ymax = float(rbboxes[i, 2] * height)
            xmax = float(rbboxes[i, 3] * width)

            if ymin < border_ymax and (rclasses[i] == tclass):
                tclasses.append(rclasses[i])
                tscores.append(rscores[i])
                tbboxes.append(rbboxes[i])
                # print("border detect", i)

        return  tclasses, tscores, tbboxes


    def conduct(self):
        # print("into behaviour state,tclass:", self._tclass)
        #print(self.ix, self.iy, self.w, self.h)

        # -- get camera img --
        k = self.context.redis_connection_img_queue.keys()
        k.sort()
        if len(k) is 0:
            print("behaviour: Queue is empty")
            time.sleep(0.1)
            return

        for img_b64 in self.context.redis_connection_img_queue.mget(k[-self.load_queue_length:]):
            img_jpg = base64.b64decode(img_b64)
            data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
            data = cv2.imdecode(data, 1)

            now_tclasses, now_tscores, now_tbboxes = self.tclass_edge_detector(data, self._tclass)
            boundingbox = self.tracker.update(data)
            boundingbox = map(int, boundingbox)
            utils.plt_tracker(data, self._tclass, boundingbox)


        self.ix = boundingbox[0]
        self.iy = boundingbox[1]
        self.w = boundingbox[2]
        self.h = boundingbox[3]

        self.move_dis = self.iy - self.ori_y
        # print ("dis_threshold", self.dis_threshold, "move dis is ", self.move_dis, "iy", self.iy, "ori_y", self.ori_y)

        if self.iy+self.h > self.dis_threshold and self.ori_y < self.dis_threshold/3:
            if self.move_dis > self.dis_threshold:
                if len(now_tclasses) is 0: # ensure can't repeat count(excepted miss detect)
                    print(self._tclass, "add one")
                    self.is_detected = 1
            else:
                pass

        if self.iy < 0: #*self.dis_threshold:
            if self.move_dis < (-self.dis_threshold-self.h/2):
                print(self._tclass, "sub one")
                self.is_detected = 1
            # elif self.move_dis < -5:
            #     if len(now_tclasses) is not 0:
            #          # print("have goods")
            #          pass
            #     else:
            #          self.is_detected = 1
            #          print("reset")
            #     pass


        pass
