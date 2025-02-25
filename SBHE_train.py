import os
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import gc
from SBHE import SBHE
from Dataset import myDataset
from metrics import metrics

gc.collect()


class CNN(object):

    def __init__(self, label_list, model_name=None):  

        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(label_list)
        self.n_label = len(label_list)
        self.trainimg_fold, self.trainimg_fold2, self.trainlabel_fold, self.trainlabel_fold2 = None, None, None, None

        self.learning_rate = 1e-2
        self.smooth = 1e-5
        self.model_name = model_name

    def train(self, model_name='SBHE', trainimg_fold=None, trainimg_fold2=None, trainlabel_fold=None,
              trainlabel_fold2=None, EPOCHS=100, batchsize=15,
              valid_rate=0.25, norm_val=0, outModel='out.pth'):
        '''
        images and lalel images should be the same file names
        :param trainlabel_fold2:
        :param trainlabel_fold:
        :param trainimg_fold2:
        :param trainimg_fold:
        :param model_name:
        :param EPOCHS: repeat times
        :param batchsize: number of input images each time
        :param valid_rate: ratio of validation images
        :param norm_val: normalization value for images, 0 means no normalization
        :param outModel: save training model
        '''
        self.trainimg_fold, self.trainimg_fold2, self.trainlabel_fold, self.trainlabel_fold2 = trainimg_fold, trainimg_fold2, trainlabel_fold, trainlabel_fold2

        if model_name == 'SBHE':
            model = SBHE(in_channels2=4, out_channels2=2, in_channels1=2, out_channels1=1)

        model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9,
                                    weight_decay=0.0005)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

        train_set, val_set = self.split_train_valid(valid_rate)
        train_numb = len(train_set)
        valid_numb = len(val_set)
        print("the number of train data is", train_numb)
        print("the number of val data is", valid_numb)
        traindata = myDataset(self.trainimg_fold, self.trainimg_fold2, train_set, self.trainlabel_fold,
                              self.trainlabel_fold2, train_set, norm_val)
        trainloader = DataLoader(traindata, batchsize, shuffle=True)
        validdata = myDataset(self.trainimg_fold, self.trainimg_fold2, val_set, self.trainlabel_fold,
                              self.trainlabel_fold2, val_set, norm_val)
        validloader = DataLoader(validdata, batchsize, shuffle=True)
        metric = metrics(self.n_label)

        best_OA = 0
        history = []
        Mseloss = nn.MSELoss()
        for epoch in range(EPOCHS):
            train_loss, train_oa, train_mIoU, train_lossT, train_lossL = 0, 0, 0, 0, 0

            train_pbar = tqdm(trainloader, dynamic_ncols=True, unit='batch')
            for batch_x, batch_x1, batch_y, batch_y1 in train_pbar:
                batch_x = batch_x.cuda()
                batch_x1 = batch_x1.type(torch.FloatTensor)
                batch_x1 = batch_x1.cuda()
                batch_y = batch_y.long().cuda()
                batch_y1 = batch_y1.type(torch.FloatTensor)
                batch_y1 = batch_y1.cuda()

                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    out1, out2 = model(batch_x, batch_x1)
                    out1 = out1.permute(0, 2, 3, 1)
                    m = out1.shape[0] * out1.shape[1] * out1.shape[2]
                    out1 = out1.resize(m, self.n_label)
                    label1 = batch_y.resize(m)

                    lossT = criterion(out1, label1)
                    lossL = Mseloss(out2, batch_y1)

                    all_loss = lossL + lossT

                    all_loss.backward()
                    optimizer.step()

                    metric.addBatch(out1.cpu().detach().numpy().argmax(axis=-1), label1.cpu())
                    oa = metric.pixelAccuracy()
                    mIoU = metric.meanIntersectionOverUnion()
                    train_pbar.set_description(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {all_loss.item():.4f}')

                    train_loss += all_loss
                    train_oa += oa
                    train_mIoU += mIoU
                    train_lossT += lossT
                    train_lossL += lossL

            lr_scheduler.step()

            val_loss, val_oa, val_mIoU, val_lossT, val_lossL = self.validation(model, validloader, criterion, Mseloss)
            val_loss = val_loss / valid_numb * batchsize
            val_oa = val_oa / valid_numb * batchsize
            val_mIoU = val_mIoU / valid_numb * batchsize
            val_lossT = val_lossT / valid_numb * batchsize
            val_lossL = val_lossL / valid_numb * batchsize
            train_loss = train_loss / train_numb * batchsize
            train_oa = train_oa / train_numb * batchsize
            train_mIoU = train_mIoU / train_numb * batchsize
            train_lossT = train_lossT / train_numb * batchsize
            train_lossL = train_lossL / train_numb * batchsize
            history.append(
                [train_loss.cpu().detach().numpy(), train_oa, train_mIoU, train_lossT.cpu().detach().numpy(),
                 train_lossL.cpu().detach().numpy(), val_loss.cpu().detach().numpy(), val_oa,
                 val_mIoU, val_lossT.cpu().detach().numpy(), val_lossL.cpu().detach().numpy()])

            print(
                "Epoch %d train loss:%f OA:%f mIoU:%f lossT:%f lossL:%f validation loss:%f OA:%f mIoU:%f lossT:%f lossL:%f" % (
                    epoch, train_loss, train_oa, train_mIoU, train_lossT, train_lossL, val_loss, val_oa, val_mIoU,
                    val_lossT, val_lossL))
            gc.collect()
            if val_oa > best_OA:
                torch.save(model.state_dict(), outModel)  
                best_OA = val_oa

    def validation(self, model, validloader, criterion, Mseloss):
        valid_loss, valid_oa, valid_mIoU, valid_lossT, valid_lossL = 0, 0, 0, 0, 0
        metric = metrics(self.n_label)
        valid_pbar = tqdm(validloader, dynamic_ncols=True, unit='batch')
        for batch_x, batch_x1, batch_y, batch_y1 in valid_pbar:
            batch_x = batch_x.cuda()
            batch_x1 = batch_x1.type(torch.FloatTensor)
            batch_x1 = batch_x1.cuda()
            batch_y = batch_y.long().cuda()
            batch_y1 = batch_y1.type(torch.FloatTensor)
            batch_y1 = batch_y1.cuda()

            with torch.set_grad_enabled(False):
                out1, out2 = model(batch_x, batch_x1)
                out1 = out1.permute(0, 2, 3, 1)
                m = out1.shape[0] * out1.shape[1] * out1.shape[2]
                out1 = out1.resize(m, self.n_label)
                label1 = batch_y.resize(m)

                lossT = criterion(out1, label1)
                lossL = Mseloss(out2, batch_y1)

                all_loss = lossL + lossT

                metric.addBatch(out1.cpu().detach().numpy().argmax(axis=-1), label1.cpu())
                oa = metric.pixelAccuracy()
                mIoU = metric.meanIntersectionOverUnion()
            valid_loss += all_loss
            valid_oa += oa
            valid_mIoU += mIoU
            valid_lossT += lossT
            valid_lossL += lossL
        return valid_loss, valid_oa, valid_mIoU, valid_lossT, valid_lossL

    def split_train_valid(self, valid_rate=0.25):
        '''
        split datasets into train and validation
        '''
        train_url = []
        train_set = []
        val_set = []
        for pic in os.listdir(self.trainimg_fold):
            train_url.append(pic)
        random.shuffle(train_url)
        total_num = len(train_url)
        val_num = int(valid_rate * total_num)
        for i in range(len(train_url)):
            if i < val_num:
                val_set.append(train_url[i])
            else:
                train_set.append(train_url[i])
        return train_set, val_set


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    modelname = 'SBHE'

    cnn = CNN(label_list=[0., 1.], model_name=modelname)
    cnn.train(model_name=modelname, trainimg_fold=r'D:\Sentinel-2', trainimg_fold2=r'D:\Sentinel-1',
              trainlabel_fold=r'D:\Footprint', trainlabel_fold2=r'D:\Building_height',
              EPOCHS=1, batchsize=1, valid_rate=0.25, norm_val=255,
              outModel=r'D:\{}_keras.h5'.format
              (modelname))
