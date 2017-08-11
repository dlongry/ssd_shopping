import threading
import redis
import time
import base64
import numpy as np
import cv2
from test_ssd_512 import detect_img
import utils
import json
import KCF
from State import State

class DecLeaveThread(threading.Thread):
    def __init__(self, context, lx_threshold=20, rx_threshold=490, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(DecLeaveThread, self).__init__(group=None, target=None, name=None, args=(), kwargs=None, verbose=None)
        self.context = context
        self.is_running = False
        self.is_close = False
        self.tracker = KCF.kcftracker(True, True, False, False)
        self.k = None
        self._tclass = -1
        self.rx_threshold = rx_threshold
        self.lx_threshold = lx_threshold

        self.face_x = float(self.context.redis_connection_webCamMapping.get("x"))
        self.face_y = float(self.context.redis_connection_webCamMapping.get("y"))
        self.face_w = float(self.context.redis_connection_webCamMapping.get("w"))
        self.face_h = float(self.context.redis_connection_webCamMapping.get("h"))
        self.basic_box = [170, 380, 150, 100]
        self.is_leave = False

    def box_transfrom(self, input_box):
        y = input_box[1]
        w = input_box[3]
        del_y = 0
        if w < 230:
            del_scale = 0.8
            if y < 60:
                del_y = 50
            elif y < 140:
                del_y = 20
            else:
                del_y = 0

        elif w < 250:
            del_scale = 0.9
            if y < 60:
                del_y = 50
            elif y < 140:
                del_y = 20
            else:
                del_y = 0

        elif w < 280:
            del_scale = 1
            if y < 60:
                del_y = 50
            elif y < 140:
                del_y = 20
            else:
                del_y = 0
        else:
            del_scale = 1.1

        print("del_scale", del_scale, "del_y", del_y)

        output_box = self.basic_box
        output_box[1] = output_box[1] +del_y
        output_box[2] = output_box[2] * del_scale
        output_box[3] = output_box[3] * del_scale

        return output_box

    def run(self):

        input_box = [self.face_x, self.face_y, self.face_w, self.face_h]
        output_box = self.box_transfrom(input_box)
       
        print("face detect: x,y,w,h", self.face_x, self.face_y, self.face_w, self.face_h)

        k = self.context.redis_connection_img_queue.keys()
        k.sort()
        if len(k) is 0:
            print("leave: Queue is empty")
            time.sleep(0.1)
            return

        for img_b64 in self.context.redis_connection_img_queue.mget(k[-1:]):
            img_jpg = base64.b64decode(img_b64)
            data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
            data = cv2.imdecode(data, 1)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            self.tracker.init(output_box, data)

        while self.is_close is not True:
            k = self.context.redis_connection_img_queue.keys()
            k.sort()
            if len(k) is 0:
                print("leave: Queue is empty")
                time.sleep(0.1)
                return

            for img_b64 in self.context.redis_connection_img_queue.mget(k[-1:]):
                img_jpg = base64.b64decode(img_b64)
                data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
                data = cv2.imdecode(data, 1)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                boundingbox = self.tracker.update(data)
                boundingbox = map(int, boundingbox)

                utils.plt_tracker(data, self._tclass, boundingbox)


                lx = boundingbox[0]
                rx = boundingbox[0] + boundingbox[2]
                end_y = boundingbox[1] + boundingbox[3]

                # print("lx,ly",lx,rx)
                # if lx < self.lx_threshold or rx > self.rx_threshold or end_y > 512:
                if lx < self.lx_threshold or rx > self.rx_threshold:
                    self.context.thread_dec_leave = None
                    self.context.redis_connection_checkout_detect.set(int(time.time() * 100000), -1)
                    time.sleep(1)
                    print("leave")
                    self.context.redis_connection_img_queue.flushdb()
                    self.is_close = True

                    # self.is_leave = True
                    self.context.current_state = State.WaitForQueue

                if boundingbox[1] < 10:
                    print("miss track,restart track")
                    self.tracker.init(self.basic_box, data)

        pass

    def mark_close(self):
        self.is_close = True

    @staticmethod
    def get_new_detect_thread(context):
        t = DecLeaveThread(context)
        t.start()
        return t
