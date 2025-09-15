import numpy as np
import cv2
import os
from sklearn.metrics import cohen_kappa_score

def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,
                                                                              n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)

def evaluation(hist, n_classes):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''

    TP = np.diag(hist)
    FP = hist.sum(0) - TP
    FN = hist.sum(1) - TP
    mIOU = TP / (TP + FP + FN)
    Precision = TP/ (TP+FP)
    Recall = TP/ (TP+FN)
    PA = TP.sum(0) / (TP.sum(0) + FP.sum(0))
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    mF1 = F1.sum(0) / n_classes
    Dice = 2*TP/(2*TP+FP+FN)

    n = np.sum(hist)
    sum_po = 0
    sum_pe = 0
    for i in range(len(hist[0])):
        sum_po += hist[i][i]
        row = np.sum(hist[i, :])
        col = np.sum(hist[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    Kappa = (po - pe) / (1 - pe)
    return mIOU, PA, mF1, Precision, Recall, Dice, Kappa # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)


def SegColor2Label(img):
    """
    img: Shape [h, w, 3]
    mapMatrix: color-> label mapping matrix,
               覆盖了Uint8 RGB空间所有256x256x256种颜色对应的label

    return: labelMatrix: Shape [h, w], 像素值即类别标签
    """
    VOC_COLORMAP = [[255, 255, 255], [0, 0, 0], ]
    #背景 [255, 0, 0] 红色
    # 小车[255,255,0]黄色
    # 防渗面[255, 255, 255]白色
    # 树 [0, 255, 0]绿色
    # 低植被[0,255, 255]青色
    # 建筑物[0, 0, 255]蓝色
    mapMatrix = np.zeros(256 * 256 * 256, dtype=np.int32)
    for i, cm in enumerate(VOC_COLORMAP):
        mapMatrix[cm[2] * 65536 + cm[1] * 256 + cm[0]] = i

    indices = img[:, :, 0] * 65536 + img[:, :, 1] * 256 + img[:, :, 2]
    return mapMatrix[indices]

def Evaluation(test_label_dir,pred_dir,name_index_map,n_classes):

    hist = np.zeros((n_classes, n_classes))
    label_path_lists = os.listdir(test_label_dir)
    for i, label_path_list in enumerate(label_path_lists):
        label = cv2.imread(os.path.join(test_label_dir,label_path_list))
        label = SegColor2Label(label)
        pred = cv2.imread(os.path.join(pred_dir,label_path_list))
        pred = SegColor2Label(pred)
        if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()),
                                                                                  os.path.join(test_label_dir,label_path_list),
                                                                                  os.path.join(pred_dir,label_path_list)))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), n_classes)
        if i > 0 and i % 10 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            mIOU, PA, F1, Precision, Recall, Dice, Kappa = evaluation(hist, n_classes)
            # print('{}'.format(str(round(np.nanmean(Recall), 4))))
            print('第{:d}步  mIOU：{}  PA：{}  F1：{} '.format(i, mIOU, PA, F1))
    mIoUs, PA , F1, Precision, Recall, Dice, Kappa = evaluation(hist, n_classes)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(n_classes):  # 逐类别输出一下mIoU值
        print('===>' + name_index_map[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> PA: ' +   str(round(np.nanmean(PA) * 100, 2)))
    print('===> F1: ' +   str(round(np.nanmean(F1) * 100, 2)))
    print('===> Precision: ' + str(round(np.nanmean(Precision) * 100, 2)))
    print('===> Recall: ' + str(round(np.nanmean(Recall) * 100, 2)))
    print('===> Dice: ' + str(round(np.nanmean(Dice) * 100, 2)))
    print('===> Kappa: '+ str(round(np.nanmean(Kappa) * 100, 2)))
    return mIoUs, PA, F1, Precision, Recall, Dice, Kappa


if __name__ == "__main__":
    test_label_dir= 'data/SHVPLI/test/labels/'
    pred_dir = 'Results/UCMNet/shvpli/'
    # pred_dir = 'Results/My_Net_2/pre/'

    name_index_map = {0: 'Insulator', 1: 'Background'}

    # name_index_map = {0:'background' , 1:'water',2:'road',3:'vegetation',4:'construction'}
    Evaluation(test_label_dir, pred_dir, name_index_map, 2)