# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:15:39 2019

@author: hefu
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 以灰色导入图像
img = cv2.imread('lena.jpg',0)   #image read be 'gray'
plt.figure(1)
plt.subplot(221);
plt.imshow(img,'gray'),plt.title('original')
plt.xticks([]),plt.yticks([])
 
# 改变图像的维度
img1 = img.reshape((img.shape[0]*img.shape[1],1))
img1 = np.float32(img1)
 
# 设定一个criteria，
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
 
# 设定一个初始类中心flags
flags = cv2.KMEANS_RANDOM_CENTERS

""" 应用K-means """
compactness,labels,centers = cv2.kmeans(img1,2,None,criteria,5,flags)
compactness_1,labels_1,centers_1 = cv2.kmeans(img1,2,None,criteria,10,flags)
compactness_2,labels_2,centers_2 = cv2.kmeans(img1,2,None,criteria,15,flags)

img2 = labels.reshape((img.shape[0],img.shape[1]))
img3 = labels_1.reshape((img.shape[0],img.shape[1]))
img4 = labels_2.reshape((img.shape[0],img.shape[1]))

plt.subplot(222)
plt.imshow(img2,'gray'),plt.title('kmeans_attempts_5')
plt.xticks([]),plt.yticks([])
plt.subplot(223)
plt.imshow(img3,'gray'),plt.title('kmeans_attempts_10')
plt.xticks([]),plt.yticks([])
plt.subplot(224)
plt.imshow(img4,'gray'),plt.title('kmeans_attempts_15')
plt.xticks([]),plt.yticks([])


""" 简单阈值  """
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # binary （黑白二值）
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # （黑白二值反转）
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)    # 得到的图像为多像素值
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)  # 高于阈值时像素设置为255，低于阈值时不作处理
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)  # 低于阈值时设置为255，高于阈值时不作处理
print(ret)
plt.figure(2)
plt.subplot(321)
plt.imshow(thresh1,'gray'),plt.title('thresh1')
plt.subplot(322)
plt.imshow(thresh2,'gray'),plt.title('thresh2')
plt.subplot(323)
plt.imshow(thresh3,'gray'),plt.title('thresh3')
plt.subplot(324)
plt.imshow(thresh4,'gray'),plt.title('thresh4')
plt.subplot(325)
plt.imshow(thresh5,'gray'),plt.title('thresh5')
plt.subplot(326)
plt.imshow(img,'gray'),plt.title('orignal')


""" 自适应阈值 """
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 第一个参数为原始图像矩阵，第二个参数为像素值上限，第三个是自适应方法（adaptive method）：
#   -----cv2.ADAPTIVE_THRESH_MEAN_C:领域内均值
#   -----cv2.ADAPTIVE_THRESH_GAUSSIAN_C:领域内像素点加权和，权重为一个高斯窗口
# 第四个值的赋值方法：只有cv2.THRESH_BINARY和cv2.THRESH_BINARY_INV
# 第五个Block size：设定领域大小（一个正方形的领域）
# 第六个参数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值，就是求得领域内均值或者加权值）
# 这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图都用一个阈值

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
plt.figure(5)
plt.subplot(211)
plt.imshow(img,'gray'),plt.title('img')
plt.subplot(212)
plt.imshow(th1,'gray'),plt.title('th1')



""" Otsu's二值化 """
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # 简单滤波
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
print(ret2)
plt.figure(4)
plt.subplot(131)
plt.imshow(img,'gray'),plt.title('img')

plt.subplot(132)
plt.imshow(th1,'gray'),plt.title('Easy filter')

plt.subplot(133)
plt.imshow(th2,'gray'),plt.title('Otsu filter')

plt.figure(3)
# 用于解决matplotlib中显示图像的中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.hist(img.ravel(), 256)
plt.title('灰度直方图')
plt.show()







