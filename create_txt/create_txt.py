'''
Description: 
创建包含图像数据的txt文件
方便使用tensorRT的C++程序读取

Author: wangdx
Date: 2021-06-29 20:06:29
LastEditTime: 2021-07-01 14:27:13
'''

import cv2
import struct


# 读取图像
img = cv2.imread('./_23_rgb.png')   # BGR
# 裁剪中间的320*320
H, W = img.shape[:2]
h, w = 320, 320
t = int((H - h) / 2)
l = int((W - w) / 2)
b = t + h
r = l + w
img = img[t:b, l:r] # (320, 320, 3)

# 逐通道写入二进制文件
# 和TensorRT的读取顺序一样
# txt每行表示图像的一行320个数字
with open('img.txt', 'w')as fp:
    for n in range(img.shape[2]):
        for r in range(img.shape[0]):
            a = ''
            for c in range(img.shape[1]):
                a += str(img[r, c, n]) + ' '
            a = a[:-1]
            fp.write(a+'\n')

print('done')




