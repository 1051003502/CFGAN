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

def paint(x,y):
    plt.title("precision")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x, y, "k-o")
    plt.ylim([0, 0.5])
    plt.show()
'''def adjustMaskVector(maskVector,m):

    itemCount=len(maskVector[0])
    for i in range(len(maskVector)):
        for j in range( m ):
            index=random.randint(0,itemCount-1)
            maskVector[i][index]=1'''
def main(trainSet,userCount,itemCount,testSet,GroundTruth,trainVector,testMaskVector,batchCount,epochCount,pro_zp):
    X=[] #画图数据的保存
    precisionList=[]
    G=model.generator(itemCount)
    D=model.discriminator(itemCount)
    G = G.cuda()
    D = D.cuda()
    criterion1 = nn.BCELoss()  # 二分类的交叉熵
    criterion2 = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

    G_step=2
    D_step=2
    batchSize_G = 32
    batchSize_D = 32
    realLabel_G = Variable(torch.ones(batchSize_G)).cuda()
    fakeLabel_G = Variable(torch.zeros(batchSize_G)).cuda()
    realLabel_D = Variable(torch.ones(batchSize_D)).cuda()
    fakeLabel_D = Variable(torch.zeros(batchSize_D)).cuda()
    ZR = []
    PM = []
    for epoch in range(epochCount): #训练epochCount次

        if(epoch%100==0):
            ZR = []
            PM = []
            for i in range(userCount):
                ZR.append([])
                PM.append([])
                ZR[i].append(np.random.choice(itemCount,pro_zp,replace=False))
                PM[i].append(np.random.choice(itemCount,pro_zp,replace=False))
        for step in range(D_step):#训练D
            #maskVector1是PM方法的体现  这里要进行优化  减少程序消耗的内存

            leftIndex=random.randint(1,userCount-batchSize_D-1)
            realData=Variable(trainVector[leftIndex:leftIndex+batchSize_D]).cuda() #MD个数据成为待训练数据

            maskVector1 = Variable(trainVector[leftIndex:leftIndex+batchSize_D]).cuda()
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
            realData = Variable(trainVector[leftIndex:leftIndex + batchSize_G]).cuda()

            maskVector2 = Variable(trainVector[leftIndex:leftIndex + batchSize_G]).cuda()
            maskVector3 = Variable(trainVector[leftIndex:leftIndex + batchSize_G]).cuda()
            for i in range(len(maskVector2)):
                maskVector2[i][PM[i+leftIndex]] = 1
                maskVector3[i][ZR[i+leftIndex]] = 1
            fakeData=G(realData)
            fakeData=fakeData*maskVector2
            g_fakeData_result=D(fakeData,realData)
            g_loss=criterion1(g_fakeData_result,realLabel_G)+0.03*criterion2(fakeData,maskVector3)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if( epoch%10==0):

            hit=0
            peopleAmount=len(GroundTruth)
            recommendAmount=10


            index=0
            for testUser in testSet.keys():
                data = Variable(trainVector[testUser]).cuda()

                result = G(data) + Variable(testMaskVector[index]).cuda()
                index+=1
                hit = hit + evaluation.computeTopNAccuracy(testSet[testUser], result, recommendAmount)

            precision=hit/(peopleAmount*recommendAmount)
            precisionList.append(precision)
            X.append(epoch)
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{}'.format(epoch, epochCount,
            d_loss.item(),
            g_loss.item(),
            hit/(peopleAmount*recommendAmount)))
            paint(X,precisionList)
    return precisionList


