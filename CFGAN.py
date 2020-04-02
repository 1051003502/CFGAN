import data
import train
if __name__ == '__main__':
    trainSet, userCount, itemCount = data.loadTrainingData("data/ml-100k/u2.base", "\t")
    userCount = 943 + 1
    itemCount = 1682 + 1
    testSet, GroundTruth = data.loadTestData("data/ml-100k/u2.test", "\t")
    userList_test = list(testSet.keys())
    trainVector, testMaskVector, batchCount = data.to_Vectors(trainSet, userCount, itemCount, userList_test, "userBased")
    train.main(trainSet,userCount,itemCount,testSet,GroundTruth,trainVector,testMaskVector,batchCount,1000,0.5,0.7,0.03)