from test_ssd_512 import detect_img
import numpy as np
import cv2
import base64
import time
import json

class BehaviourDetection:
    def __init__(self, context, load_queue_length=5):
        self.load_queue_length = load_queue_length
        self.context = context
        self.is_detected = False
        self._tclass = None
        self._det_obj_flag = 0
        self._divide_flag = 0
        self._ori_place = 0
        self._ignore_count = 0
        pass

    def rItem_detector(self, img, t_class):
        prop_border = self.context.edge_detector.prop_border
        rclasses, rscores, rbboxes = detect_img(img)
        for i in range(rclasses.shape[0]):
            if rclasses[i] == t_class:
                self._det_obj_flag = 1
                if rbboxes[i, 0] > prop_border:  # bboxes[i,0] is ymin > means out of border
                    self._divide_flag = 1
        pass

    def conduct(self):
        print("into behaviour state,tclass:", self.context.edge_detector.now_tclasses[0], "ori_place:", self.context.edge_detector.ori_places[0])
        self._ori_place = self.context.edge_detector.ori_places[0]
        self._tclass = self.context.edge_detector.now_tclasses[0]


        #
        # if self._ori_place == 1:
        #     print(self._tclass, "substract one")
        #     self.context.redis_connection_result_queue.set(int(time.time() * 100000), json.dumps({'operator': '-', 'item_id': '1'}))
        # else:
        #     print(self._tclass,"add one")
        # time.sleep(0.5)
        # self.is_detected = True
        '''
        k = self.context.redis_connection_img_queue.keys()
        k.sort()
        if len(k) is 0:
            print("EdgeDetection: Queue is empty")
            return

        for img_b64 in self.context.redis_connection_img_queue.mget(k[-self.load_queue_length:]):
            img_jpg = base64.b64decode(img_b64)
            data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
            data = cv2.imdecode(data, 1)
            self.rItem_detector(data, self._tclass)

        if self._det_obj_flag == 0:
            # self._ignore_count = self._ignore_count + 1
            # if self._ignore_count > 5:
            if self._ori_place == 0:  # may be ignoring object
                print("do nothing")
                self.is_detected = True
            else:
                print(self._tclass, "substract one")
                self.context.redis_connection_result_queue.set(int(time.time() * 100000), json.dumps({'operator': '-', 'item_id': '1'}))
                # set to redis
                self.is_detected = True

        else:
            if self._divide_flag != 0:
                if self._ori_place == 0:
                    print(self._tclass, "add one")
                    self.context.redis_connection_result_queue.set(int(time.time() * 100000), json.dumps({'operator': '+', 'item_id': '1'}))
                    # set to redis
                    self.is_detected = True
                else:
                    #can't happend
                    self._divide_flag=0
                    self._det_obj_flag=0
                    print("down do nothing")

            else:
                print("no leave")

            '''
        pass
