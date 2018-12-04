import CNNlib as cnnlib
from keras.models import Sequential
import numpy as np

Xtrain = None
Ytrain = None
model = None

CNN = cnnlib.CNNmodel('TestData')
CNN.connectpath()
CNN.readData()
#print('picarray:',len(CNN.getarray()))
#print('change:',CNN.changetype())
CNN.setDictionary()
#print('dictionary:',CNN.getDictionary())
CNN.jsonpacking()
#CNN.loadjson()
CNN.normalization()
CNN.one_hot()

Xtrain = CNN.getarray()
Ytrain = CNN.getlabels()

Xtrain = np.expand_dims(Xtrain,3)

print(Xtrain.shape)

model = Sequential()
model = CNN.createmodel(model)

print(CNN.getsummary(model))

model = CNN.train(model,Xtrain,Ytrain)

print(CNN.getevaluate(model,Xtrain,Ytrain))

path = "D:\\Unlimited-Gold-Work\\kerasCNN\\data\\TestData\\moto\\moto3.jpg"

ans = CNN.getpredict(model,path)

CNN.answer(ans)