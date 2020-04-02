import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import random
import evaluation
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def paint(x,y1,y2,y3):
    plt.title("CFGAN")
    plt.xlabel('epoch')
    #plt.ylabel('')
    plt.plot(x, y2, "k-o", color='red', label='recall', markersize='0')
    plt.plot(x, y1, "k-o",color='black',label='precision',markersize =  '0' )#List,List,List

    plt.plot(x, y3, "k-o",color='green',label='ndcg',markersize =  '0')
    plt.ylim([0, 0.5])
    plt.legend()  # 图例
    plt.rcParams['lines.linewidth'] = 1
    plt.show()


def main(trainSet,userCount,itemCount,testSet,GroundTruth,trainVector,testMaskVector,batchCount,epochCount,pro_ZR,pro_PM,arfa):
    X=[] #画图数据的保存
    precisionList=[]
    recallList=[]
    ndcgList=[]
    G=model.generator(itemCount)
    D=model.discriminator(itemCount)
    G = G.cuda()
    D = D.cuda()
    criterion1 = nn.BCELoss()  # 二分类的交叉熵
    criterion2 = nn.MSELoss(size_average=False)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

    G_step=2
    D_step=2
    batchSize_G = 32
    batchSize_D = 32
    realLabel_G = (torch.ones(batchSize_G)).cuda()
    fakeLabel_G = (torch.zeros(batchSize_G)).cuda()
    realLabel_D = (torch.ones(batchSize_D)).cuda()
    fakeLabel_D = (torch.zeros(batchSize_D)).cuda()
    ZR = []
    PM = []
    for epoch in range(epochCount): #训练epochCount次

        if(epoch%1==0):
            ZR = []
            PM = []
            for i in range(userCount):
                ZR.append([])
                PM.append([])
                ZR[i].append(np.random.choice(itemCount,int(pro_ZR*itemCount),replace=False))
                PM[i].append(np.random.choice(itemCount,int(pro_PM*itemCount),replace=False))
        for step in range(D_step):#训练D
            #maskVector1是PM方法的体现  这里要进行优化  减少程序消耗的内存

            leftIndex=random.randint(1,userCount-batchSize_D-1)
            realData=(trainVector[leftIndex:leftIndex+batchSize_D]).cuda() #MD个数据成为待训练数据

            maskVector1 = (trainVector[leftIndex:leftIndex+batchSize_D]).cuda()
            for i in range(len(maskVector1)):
                maskVector1[i][PM[leftIndex+i]]=1
            Condition=realData#把用户反馈数据作为他的特征 后期还要加入用户年龄、性别等信息
            realData_result=D(realData,Condition)
            d_loss_real=criterion1(realData_result,realLabel_D)

            fakeData=G(realData)
            fakeData=fakeData*maskVector1
            fakeData_result=D(fakeData,realData)
            d_loss_fake=criterion1(fakeData_result,fakeLabel_D)

            d_loss=d_loss_real+d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
        for step in range(G_step):#训练G0
            #调整maskVector2\3
            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            realData = (trainVector[leftIndex:leftIndex + batchSize_G]).cuda()

            maskVector2 = (trainVector[leftIndex:leftIndex + batchSize_G]).cuda()
            maskVector3 = (trainVector[leftIndex:leftIndex + batchSize_G]).cuda()
            for i in range(len(maskVector2)):
                maskVector2[i][PM[i+leftIndex]] = 1
                maskVector3[i][ZR[i+leftIndex]] = 1
            fakeData=G(realData)
            g_loss2=arfa * criterion2(fakeData, maskVector3)
            fakeData=fakeData*maskVector2
            g_fakeData_result=D(fakeData,realData)
            g_loss1=criterion1(g_fakeData_result,realLabel_G)
            g_loss=g_loss1+g_loss2
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if( epoch%10==0):

            hit=0
            peopleAmount=len(GroundTruth)
            recommendAmount=10
            index=0
            precisions=0
            recalls=0
            ndcgs=0
            for testUser in testSet.keys():
                data = (trainVector[testUser]).cuda()

                result = G(data) + (testMaskVector[index]).cuda()
                index+=1
                precision,recall,ndcg=evaluation.computeTopNAccuracy(testSet[testUser], result, recommendAmount)
                precisions+=precision
                recalls+=recall
                ndcgs+=ndcg

            precisions /= peopleAmount
            recalls /= peopleAmount
            ndcgs /= peopleAmount

            precisionList.append(precisions)
            recallList.append(recalls)
            ndcgList.append(ndcgs)

            X.append(epoch)
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{},recall:{},ndcg:{}'.format(epoch, epochCount,
            d_loss.item(),
            g_loss.item(),
            precisions,recalls,ndcgs))
            paint(X,precisionList,recallList,ndcgList)
    return precisionList


