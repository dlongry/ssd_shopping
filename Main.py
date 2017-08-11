from State import State
from EdgeDetection import EdgeDetection
from QueueChecker import QueueChecker
import redis
import numpy as np
from BehaviourDetection import BehaviourDetection
from DoubleCheck import DoubleCheck
from DecLeaveThread import DecLeaveThread

import cv2


class Main:
    def __init__(self):
        self.current_state = State.WaitForQueue
        self.edge_detector = None
        self.behaviour_detector = None
        self.doublecheck =None
        self.redis_connection_img_queue = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.redis_connection_result_queue = redis.StrictRedis(host='localhost', port=6379, db=1)
        self.redis_connection_checkout_detect = redis.StrictRedis(host='localhost',port=6379,db=3)
        self.redis_connection_webCamMapping = redis.StrictRedis(host='localhost',port=6379,db=4)
        self.queue_checker = None
        self.canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        self.thread_dec_leave = None

    def __del__(self):
        if self.thread_dec_leave is not None:
            self.thread_dec_leave.mark_close()

    def execute_and_update_state(self):
        #print(self.current_state)
        if self.current_state == State.WaitForQueue:
            self.queue_checker = QueueChecker(self)
            self.current_state = State.Start

        if self.current_state == State.Start:
            if self.queue_checker.is_go():
                self.edge_detector = EdgeDetection(self)
                self.current_state = State.EdgeDetection
                # #add init DecLeaveThread
                if self.thread_dec_leave is None:
                    self.thread_dec_leave = DecLeaveThread.get_new_detect_thread(self)

        elif self.current_state == State.EdgeDetection:
            # TODO: how to detect edge
            # if False:
            if self.edge_detector.is_detected:  # xTODO: edge detection success:
                self.behaviour_detector = BehaviourDetection(self)
                self.current_state = State.BehaviourDetection
            else:
                self.edge_detector.conduct()
        elif self.current_state == State.BehaviourDetection:
            # TODO: how to detect behaviour
            if self.behaviour_detector.is_detected == 1:  # TODO: single item bought, may be finished
                self.current_state = State.Start

            elif self.behaviour_detector.is_detected == 2:
                self.doublecheck = DoubleCheck(self)
                self.current_state = State.BehaviourDoubleCheck
            else:
                self.behaviour_detector.conduct()
        elif self.current_state == State.BehaviourDoubleCheck:
            if self.doublecheck.is_detected == 1:
                self.current_state = State.BehaviourDetection
            elif self.doublecheck.is_detected == 2:
                self.current_state = State.Start
            else:
                self.doublecheck.conduct()

            pass
        pass

    def render(self):
        pass
