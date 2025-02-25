import numpy as np


class metrics(object):

    def __init__(self, n_label):
        self.n_label = n_label
        self.confusionMatrix = np.zeros((self.n_label,) * 2)

    def pixelAccuracy(self):  # OA
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        acc = round(acc, 5)
        return acc

    def classPixelAccuracy(self):  # PA
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        mIoU = round(mIoU, 4)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.n_label)
        label = self.n_label * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.n_label ** 2)
        confusionMatrix = count.reshape(self.n_label, self.n_label)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.n_label, self.n_label))

    def recall(self):
        classRecall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classRecall
