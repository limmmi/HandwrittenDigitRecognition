import numpy as np
import cv2
import torch
from CNN_MODEL import CNN_Net
import CNN_2


def pretreatment(file_name: str):
    image = cv2.imread(file_name)

    # 转灰度
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = 255 - img

    # 二值化 阈值, 图象
    ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 图象分块
    img.dtype = np.uint8
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    stats = stats[1:,]
    if retval == 1:
        return 'ERROR'
    # 筛选分块信息
    save_stats = []
    for i in range(len(stats)):
        if stats[i][2] < 10 and stats[i][3] < 10:
            pass
        else:
            save_stats.append(stats[i])
    save_stats = np.array(save_stats)

    many_img_s = to_scan(save_stats)
    list_x = []
    for i in range(len(many_img_s)):
        # # 图像居中宽展
        max_line = max(many_img_s[i][0][2], many_img_s[i][0][3])
        rate = 1.74

        line = int(rate * max_line)
        new_image = np.zeros((line, line))

        y0 = many_img_s[i][0][1]
        y1 = many_img_s[i][0][1] + many_img_s[i][0][3]
        x0 = many_img_s[i][0][0]
        x1 = many_img_s[i][0][0] + many_img_s[i][0][2]
        img2 = img[y0:y1, x0:x1]
        left_top = ((line - many_img_s[i][0][2]) // 2, (line - many_img_s[i][0][3]) // 2)
        right_bottom = ((line + many_img_s[i][0][2]) // 2, (line + many_img_s[i][0][3]) // 2)
        new_image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] = img2
        # cv2.imshow('new_img', new_image)
        # cv2.waitKey()
        x = cv2.resize(new_image, (28, 28), interpolation=cv2.INTER_AREA)

        x[np.where(x != 0)] = 1
        # Normalize
        # x = x - 0.1307
        # x = x / 0.3081

        x = x.reshape((1, 28, 28))
        list_x.append(x)
    x = np.array(list_x)
    x1 = torch.tensor(x, dtype=torch.float)

    return x1


# 确定合适的二值化阈值
def make_thresh(img):
    thresh = 0
    min_retval = 300
    r = 300
    for i in range(15, 255):
        # 二值化 阈值, 图象
        ret, img_x = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)

        img_x.dtype = np.uint8
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img_x, connectivity=8)
        if retval < r:
            thresh = i
            min_retval = retval
            r = retval
    return thresh


def to_scan(stats):
    distance = 50
    list_stats = []
    record = [0]*len(stats)
    for i in range(stats.shape[0]):
        for j in range(i+1, stats.shape[0]):
            if abs(stats[i][0]-stats[j][0])+abs(stats[i][1]-stats[j][1]) < distance:
                list_stats.append([stats[i], stats[j]])
                record[i] = 1
                record[j] = 1
        if record[i] == 0:
            list_stats.append([np.array(stats[i])])
    for i in range(len(list_stats)):
        if len(list_stats[i]) > 1:
            left = 600
            right = 0
            top = 600
            bottom = 0
            s = 0
            for j in range(len(list_stats[i])):
                s += list_stats[i][j][-1]
                if list_stats[i][j][0] < left:
                    left = list_stats[i][j][0]
                if list_stats[i][j][0] + list_stats[i][j][2] > right:
                    right = list_stats[i][j][0] + list_stats[i][j][2]
                if list_stats[i][j][1] < top:
                    top = list_stats[i][j][1]
                if list_stats[i][j][1] + list_stats[i][j][3] > bottom:
                    bottom = list_stats[i][j][1] + list_stats[i][j][3]
            list_stats[i] = np.array([[left, top, right - left, bottom - top, s]])

    ans = np.array(list_stats)
    batch_size = ans.shape[0]

    arr = ans[:, 0]
    arr = arr[np.argsort(arr[:, 0])]
    ans = np.array([arr])
    ans = ans.reshape((batch_size, 1, 5))
    return ans


if __name__ == '__main__':
    pretreatment('canvas.jpg')