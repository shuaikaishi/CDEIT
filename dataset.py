import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import os
from matplotlib import pyplot as plt
# from kymatio.torch import Scattering1D
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EITdataset(Dataset):
    """
    return Gaussian-newton(condition), ground truth
    """

    def __init__(self, data_file_path, modelname=None,dataset='simulate'):
        modelnameList = ['ImprovedLeNet', 'CNNEIM', 'SADBnet', 'SAHFL', 'EcNet', 'DHUnet','DEIT']
        assert modelname in modelnameList, \
            'modelname should be in ' + str(modelnameList)
        self.filenames = os.listdir(data_file_path)
        
        self.filenames.sort()
        # self.filenames = self.filenames[0:1]
        self.data_file_path = data_file_path
        self.modelname = modelname
        # self.scattering = Scattering1D(J=3, Q=8, shape=208)
        if dataset == 'simulate':
            self.voltage=1
            self.current=1
        elif dataset == 'data2017':
            self.voltage=  1.040856e3  # V2/V1
            self.current=2    # I2/I1
        elif dataset == 'data2023':
            self.voltage=1978  # V2/V1
            self.current=2    # I2/I1
        elif dataset == 'data2024':
            self.voltage=3022e4  # V2/V1
            self.current=8    # I2/I1
        else:
            self.voltage = 1
            self.current = 1


    def toEIM(self, voltage):
        '''

        Args:
            voltage: [1,16,13]

        Returns:
            eim:[1,16,16] EIM map

        '''

        num = 16  # num of electrodes
        eim = torch.zeros(1, num, num)

        # new_matrix = np.zeros((16, 16), dtype=int)

        # 遍历每一行
        for i in range(num):
            # 确定本行零的位置
            zero_positions = [(i + j) % num for j in range(3)]

            row = zero_positions[1]

            # 确定数据的开始位置
            # start_pos = (row + 1) %  num

            # 填充非零元素
            non_zero_index = 0
            for j in range(num):
                # start_pos=
                idx = (j) % num

                if idx not in zero_positions:
                    eim[:, row, idx] = voltage[:, row, non_zero_index]
                    non_zero_index += 1

        return eim

    def EIMtoEIV(self, voltage):
        '''

        Args:
            voltage: [1,16,16]

        Returns:
            eiv: [104]

        '''
        num = 16
        idx = 0
        eiv = torch.zeros([num * (num - 3) // 2])
        for i in range(num):
            for j in range(num):
                if j > i + 1:
                    if i == 0 and j == num - 1:
                        # print(i, j)
                        continue
                    # print(i, j, idx)
                    eiv[idx] = voltage[0, i, j]
                    idx += 1
        return eiv

    def norm(self, ys):
        # data_file_path
        mean = torch.load(self.data_file_path+'../mean.pth')
        std = torch.load(self.data_file_path+'../std.pth')
        assert self.modelname in ['ImprovedLeNet', 'CNNEIM', 'SADBnet', 'SAHFL','DEIT']
        # print(ys.shape,mean.shape,std.shape)

        return (ys - mean) / std

    def __getitem__(self, index):

        data_file = self.data_file_path + self.filenames[index]
        modelname = self.modelname
        # x_inv = self.xs_inv[index]
        # x = self.xs[index]
        # y = self.ys[index]
        d = np.load(data_file)
        xs_all = d['xs']
        # xs_inv_all = d['xs_gn']
        xs_TR_all = d['TR']
        # xs_TR_all = d['xs_gn']
        ys_all = d['ys'][:, 0]/self.voltage
        xs = torch.from_numpy(xs_all).float().unsqueeze(axis=0)

        if modelname == 'ImprovedLeNet':
            ys = torch.from_numpy(ys_all).float()
            ys = ys.reshape([1, 16, 13])
            ys = self.norm(ys)
        elif modelname == 'CNNEIM' or modelname == 'DEIT':
            ys = torch.from_numpy(ys_all).float()
            ys = ys.reshape([1, 16, 13])
            ys = self.norm(ys)



            ys_st = ys#self.scattering(ys.reshape([-1,1])[:, 0])[0]
            # self.xs = torch.from_numpy(xs_all).float().unsqueeze(axis=0)
            ys = self.toEIM(ys)
            # ys=ys+ys.permute(0,2,1)
            # ys=ys*0.5
        # elif modelname == ''
        elif modelname == 'SADBnet' or modelname == 'SAHFL':
            ys = torch.from_numpy(ys_all).float()
            ys = ys.reshape([1, 16, 13])
            ys = self.norm(ys)


            # self.xs = torch.from_numpy(xs_all).float().unsqueeze(axis=0)
            ys = self.toEIM(ys)
            ys = self.EIMtoEIV(ys)


        elif modelname == 'EcNet' or modelname == 'DHUnet':
            ys = torch.from_numpy(xs_TR_all).float().unsqueeze(axis=0)*self.voltage/self.current
            # self.xs = torch.from_numpy(xs_all).float().unsqueeze(axis=0)

        # xs_TR = self.xs_TR[index]
        return ys,ys_st, xs  # x_inv, x, y, xs_TR

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    modelname = ['ImprovedLeNet', 'CNNEIM', 'SADBnet', 'SAHFL', 'EcNet', 'DHUnet']
    # data_file_list = []
    datapath = '/dpc/kuin0055/eitssk/data/dataDiscrete'
    path =  datapath +'/test/'
    # for i in os.listdir(path):
    #     data_file_list.append(path + i)

    modelname = modelname[0]
    data = EITdataset(path, modelname)
    print(modelname)
    # mean = inp.mean()

    a = DataLoader(data, batch_size=500)
    # import torch.nn as nn
    #
    # ln = nn.LayerNorm(104 * 2)
    #
    total = []
    for data in a:
        total.append(data[0])
        print(data[0].shape)
        print(data[0].mean(0),data[0].std(0))
        # break
    pred=torch.cat(total,0)
    pred=pred.squeeze()
    print(pred.shape,pred.dtype)
    # import  scipy.io as sio
    # sio.savemat('GN.mat',
    #             {'pred': pred.cpu()})
    # sio.savemat(weightpath +     'GT.mat',
    #             {'GT': GT.cpu()})
    # print('out', pred.shape)
    total = torch.cat(total, dim=0)
    mean = torch.mean(total, 0)
    std = torch.std(total, 0)
    # torch.save(mean, datapath +'/mean.pth')
    # torch.save(std, datapath +'/std.pth')
    # mean = torch.load('data/mean.pth')
    # std = torch.load('data/std.pth')
    # print(mean.shape, mean.dtype)
    # plt.plot(mean[0].reshape(208, 1))
    # plt.show()
    # plt.plot(std[0].reshape(208, 1))
    # plt.show()

    # print(data[0][0].shape)

    # res = (data[0][1] - mean) / std
    # print(data[0][0].shape)
    # plt.plot(data[0][0].reshape(208))
    # plt.plot(data[0][0])
    # plt.imshow(data[1][0, 0])
    # plt.colorbar()
    # plt.show()
    # o = ln(inp.unsqueeze(1))
    # plt.plot(o.detach()[0,0])
    # plt.show()
    # print(o.mean(), o.std())
    # print(mean, std)
    #     print(data[0].shape, data[1].shape)
    #     print(o.shape)
    #     print(o)
