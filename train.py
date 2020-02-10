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
def adjustMaskVector(maskVector,m):

    itemCount=len(maskVector[0])
    for i in range(len(maskVector)):
        for j in range(  int(itemCount*m)  ):
            index=random.randint(0,itemCount-1)
            maskVector[i][index]=1
def main(trainSet,userCount,itemCount,testSet,GroundTruth,trainVector,testMaskVector,batchCount,epochCount,pro_zm):
    X=[]
    precisionList=[]
    G=model.generator(itemCount)
    D=model.discriminator(itemCount)
    D = D.cuda()
    G = G.cuda()
    criterion = nn.BCELoss()  # 二分类的交叉熵

    d_optimizer = torch.optim.SGD(D.parameters(), lr=0.005)  # 原作者的0.003训练不出来结果

    g_optimizer = torch.optim.SGD(G.parameters(), lr=0.005)
    maskVector=copy.deepcopy(trainVector)
    adjustMaskVector(maskVector,pro_zm)
    for epoch in range(epochCount): #训练上个epochCount次
        #生成M(t)

        #训练D
        MD=16
        realLabel=Variable(torch.ones(MD)).cuda()
        fakeLabel=Variable(torch.zeros(MD)).cuda()
        leftIndex=random.randint(1,userCount-MD-1)
        realData=Variable(trainVector[leftIndex:leftIndex+MD]).cuda() #MD个数据成为待训练数据

        maskVector1 = Variable(maskVector[leftIndex:leftIndex+MD]).cuda()


        realData_result=D(realData,realData)
        d_loss_real=criterion(realData_result,realLabel)

        fakeData=G(realData)
        fakeData=fakeData*maskVector1
        fakeData_result=D(fakeData,realData)
        d_loss_fake=criterion(fakeData_result,fakeLabel)

        d_loss=d_loss_real+d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
            #Sample minibatch of MD users
            #Get their real purchase vector {r1,r2,...r md}
            #Generate fake purchase vector  {1,2,...      }
            
        #训练G
        leftIndex = random.randint(1, userCount - MD - 1)
        realData = Variable(trainVector[leftIndex:leftIndex + MD]).cuda()  # 50个数据成为待训练数据

        maskVector2 = Variable(maskVector[leftIndex:leftIndex + MD]).cuda()


        fakeData=G(realData)
        fakeData=fakeData*maskVector2
        g_fakeData_result=D(fakeData,realData)
        g_loss=criterion(g_fakeData_result,realLabel)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if(epoch%50==0):adjustMaskVector(maskVector,pro_zm)
        if( epoch%100==0):

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


