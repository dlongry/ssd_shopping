# -*- coding: utf-8 -*-

import base64
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import redis
import json

import sys
sys.path.append('./SSD-Tensorflow-512')

from test_ssd_512 import detect_img
from notebooks import visualization

PROP_BORDER = 0.1

fig = plt.figure(figsize=(10, 10))

Flag=0


def main():
    global r2
    # Test on some demo image and visualize output.
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    r2 = redis.StrictRedis(host='localhost', port=6379, db=2)
    rclasses, rscores, rbboxes = detect_img(cv2.imread('./SSD-Tensorflow-512/demo2/Astick_00059.jpg'))
    while True:
        k = r.keys()
        k.sort()
        if len(k) is 0:
            time.sleep(1)
            print("waiting for queue")
            continue

        for img_b64 in r.mget(k[-10:]):
            t = time.time()
            img_jpg = base64.b64decode(img_b64)
            # data = np.asarray(bytearray(img_jpg), dtype=np.uint8).reshape(512,512,3)[:,:,::-1]
            data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
            data = cv2.imdecode(data, 1)

            tclass,tscore,tbbox = detector_border(data)

            if tclass==[2]: # detected border
                Flag=1
            #get origin place of object:
                t_oriPla = get_ori_place(tclass,tbbox)
                r2.flushdb()
                print(time.time(), t_oriPla)

                while True:
                    k = r.keys()
                    k.sort()
                    if len(k) is 0:
                        time.sleep(1)
                        continue

                    det_obj_flag=0
                    divide_flag=0
                    for img_b64 in r.mget(k[-1:]):
                        img_jpg = base64.b64decode(img_b64)
                        data = np.asarray(bytearray(img_jpg), dtype=np.uint8)
                        data = cv2.imdecode(data, 1)
                        rclasses, rscores, rbboxes = detect_img(data)
                        visualization.plt_bboxes_cv(fig, data, rclasses, rscores, rbboxes)

                        for i in range(rclasses.shape[0]):
                            if rclasses[i] == tclass[0]:
                                det_obj=1
                                if rbboxes[i, 0] > PROP_BORDER: #bboxes[i,0] is ymin > means out of border
                                    divide_flag=1
                                break

                    if det_obj_flag == 0:
                        if t_oriPla == 0:
                            print("do nothing")
                        else:
                            print("tclass back one")
                    else:
                        if divide_flag==0:
                            continue
                        else:
                            if t_oriPla ==0:
                                print("tclass add one")

                            else:
                                print("do nothing")
                    break

            #print(1 / (time.time() - t), 'fps')


def get_ori_place(tclass,tbbox):
    k = r2.keys()
    if len(k) is 0:
        return 0
    else:
        for results in r2.mget(k):
            try:
                d = json.loads(results)
            except:
                print('aa')
                pass
            nclass = d['rclasses']
            nsore = d['rscores']
            nbbox= d['rbboxes']
            if nclass != [2]: #tclass:
                continue
            else:  
                return 1  #future:add ROI
    return 0

def detector_border(img):
    """
    detect goods whether touch border
    """
    tclasses = []
    tscores = []
    tbboxes = []

    height = np.shape(img)[0]
    width = np.shape(img)[1]
    border_ymax = height * PROP_BORDER

    rclasses, rscores, rbboxes = detect_img(img)
    #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    d = {
        'rclasses': rclasses.tolist(),
        'rscores': rscores.tolist(),
        'rbboxes': rbboxes.tolist()
    }




    # visualization.plt_bboxes_cv(fig, img, rclasses, rscores, rbboxes)

    # 保存检测结果2s

    for i in range(rclasses.shape[0]):
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)
	        
        if ymin < border_ymax:
            tclasses.append(rclasses[i])
            tscores.append(rscores[i])
            tbboxes.append(rbboxes[i])
            print("border detect",i)
        else:
            if rclasses.shape[0] is not 0 and Flag ==0:
                r2.set(int(time.time()*100000), json.dumps(d), px=100)



    return tclasses, tscores, tbboxes


if __name__ == '__main__':
    #img = mpimg.imread('./SSD-Tensorflow-512/demo2/OreoCookie_00154.jpg')
    #rtn = process_img(img)
    #print(rtn)
    main()
