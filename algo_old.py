# -*- coding: utf-8 -*-

import base64
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import redis

import sys
sys.path.append('./SSD-Tensorflow-512')

from test_ssd_512 import detect_img
from notebooks import visualization

PROP_BORDER = 0.2

fig = plt.figure(figsize=(10, 10))

def main():
    # Test on some demo image and visualize output.

    r = redis.StrictRedis(host='localhost', port=6379, db=0)
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

            process_img(data)

            #print(1 / (time.time() - t), 'fps')


def process_img(img):
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
    visualization.plt_bboxes_cv(fig, img, rclasses, rscores, rbboxes)

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

    return tclasses, tscores, tbboxes


if __name__ == '__main__':
    #img = mpimg.imread('./SSD-Tensorflow-512/demo2/OreoCookie_00154.jpg')
    #rtn = process_img(img)
    #print(rtn)
    main()
