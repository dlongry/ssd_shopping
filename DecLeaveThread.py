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

        self.face_x = None
        self.face_y = None
        self.face_w = None
        self.face_h = None
        self.basic_box = [170, 380, 150, 100]
        self.is_leave = False

    def box_transfrom(self, input_box):
        y = input_box[1]
        w = input_box[3]
        del_y = 0
        if w < 115:
            del_scale = 0.8
            if y < 30:
                del_y = 25
            elif y < 70:
                del_y = 10
            else:
                del_y = 0

        elif w < 125:
            del_scale = 0.9
            if y < 30:
                del_y = 25
            elif y < 70:
                del_y = 10
            else:
                del_y = 0

        elif w < 140:
            del_scale = 1
            if y < 30:
                del_y = 25
            elif y < 70:
                del_y = 10
            else:
                del_y = 0
        else:
            del_scale = 1.1

        print("del_scale", del_scale, "del_y", del_y)

        output_box = self.basic_box
        output_box[1] = output_box[1] + del_y
        output_box[2] = output_box[2] * del_scale
        output_box[3] = output_box[3] * del_scale

        return output_box

    def run(self):
        while self.context.redis_connection_webCamMapping.get("x") is None:
            print(self.context.redis_connection_webCamMapping.get("x"))
            time.sleep(0.1)

        self.face_x = float(self.context.redis_connection_webCamMapping.get("x"))
        self.face_y = float(self.context.redis_connection_webCamMapping.get("y"))
        self.face_w = float(self.context.redis_connection_webCamMapping.get("w"))
        self.face_h = float(self.context.redis_connection_webCamMapping.get("h"))

        input_box = [self.face_x, self.face_y, self.face_w, self.face_h]
        output_box = self.box_transfrom(input_box)

        print("face detect: x,y,w,h", self.face_x, self.face_y, self.face_w, self.face_h)

        k = self.context.redis_connection_img_queue.keys()
        k.sort()
        while len(k) is 0:
            print("leave: Queue is empty")
            time.sleep(0.1)


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
                # print("leave: Queue is empty")
                time.sleep(0.1)
                continue

            for img_b64 in self.context.redis_connection_img_queue.mget(k[-1:]):
                img_jpg = base64.b64decode(img_b64)
                data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
                data = cv2.imdecode(data, 1)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                boundingbox = self.tracker.update(data)
                boundingbox = map(int, boundingbox)

                # utils.plt_tracker(data, self._tclass, boundingbox)

                lx = boundingbox[0]
                rx = boundingbox[0] + boundingbox[2]
                end_y = boundingbox[1] + boundingbox[3]

                # print("lx,ly",lx,rx)
                # if lx < self.lx_threshold or rx > self.rx_threshold or end_y > 512:
                if lx < self.lx_threshold or rx > self.rx_threshold:
                    self.context.thread_dec_leave = None
                    self.context.redis_connection_checkout_detect.set(int(time.time() * 100000), -1)
                    self.context.redis_connection_webCamMapping.flushdb()
                    time.sleep(1)
                    print("leave")
                    self.context.redis_connection_img_queue.flushdb()
                    self.is_close = True
                    self.context.current_state = State.Start

                if self.context.redis_connection_reset.get('reset') is not None:
                    print("reset_flag:",self.context.redis_connection_reset.get('reset'))
                    if self.context.redis_connection_reset.get('reset') == 'True':
                        self.context.thread_dec_leave = None
                        self.context.redis_connection_checkout_detect.set(int(time.time() * 100000), -1)
                        self.context.redis_connection_reset.flushdb()
                        self.context.redis_connection_webCamMapping.flushdb()
                        time.sleep(1)
                        print("restart")
                        self.context.redis_connection_img_queue.flushdb()
                        self.is_close = True
                        self.context.current_state = State.Start


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
