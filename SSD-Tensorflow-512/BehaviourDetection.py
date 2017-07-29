from test_ssd_512 import detect_img
import numpy as np
import cv2
import base64
import time
import json
import KCF
import utils


class BehaviourDetection:
    def __init__(self, context, load_queue_length=1, img_w=512, img_h=512):
        self.load_queue_length = load_queue_length
        self.context = context
        self.img_w = img_w
        self.img_h = img_h
        self.is_detected = False
        self._tclass = self.context.edge_detector.now_tclasses[0]
        self.tracker = KCF.kcftracker(True, True, True,
                                      False)  # hog, fixed_window, multiscale, lab(True, True, False,lab)
        self.ix = int(self.context.edge_detector.now_tbboxesr[0][1] * img_w)
        self.iy = int(self.context.edge_detector.now_tbboxesr[0][0] * img_h)  # init (x0,y0)
        self.w = int(self.context.edge_detector.now_tbboxesr[0][3] * img_w) - int(
            self.context.edge_detector.now_tbboxesr[0][1] * img_w)
        self.h = int(self.context.edge_detector.now_tbboxesr[0][2] * img_h) - int(
            self.context.edge_detector.now_tbboxesr[0][0] * img_h)
        self._ori_place = self.context.edge_detector.ori_places[0]

        self.tracker.init([self.ix, self.iy, self.w, self.h], self.context.edge_detector.last_frame)
        pass

    def conduct(self):
        print("into behaviour state,tclass:", self._tclass, "ori_place:", self._ori_place)
        print(self.ix, self.iy, self.w, self.h)

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

            boundingbox = self.tracker.update(data)
            boundingbox = map(int, boundingbox)
            utils.plt_tracker(data, self._tclass, boundingbox)
            self.ix = boundingbox[0]
            self.iy = boundingbox[1]
            self.w = boundingbox[2]
            self.h = boundingbox[3]

        pass
