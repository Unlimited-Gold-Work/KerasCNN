import json
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

class CNNmodel():

	'''
		初始化建構子
	'''
	def __init__(self,setpath,allpath='data'):
		self.allpath=allpath		#紀錄最上層路徑(固定)
		self.setpath=setpath		#記錄總圖片集路徑
		self.connpath=None			#紀錄相加後的路徑
		self.kinds=[]				#紀錄種類名
		self.patharray=[]			#紀錄路徑名
		self.picarray=[]			#記錄圖片資料
		self.piccount=[]			#記錄單一圖片集的張數
		self.labels=[]				#紀錄labels
		self.dictionary=[]			#紀錄dict

	'''
		回傳圖片集資料夾的list
	'''
	def connectpath(self):
		# ./data/[setpath]
		self.connpath = os.path.join(self.allpath,self.setpath)
		print('connpath:',self.connpath)
		# ./data/[setpath]底下的資料夾與檔案
		filelist = os.listdir(self.connpath)
		for file in filelist:
			print('file:',file)
			filedir = os.path.join(self.connpath,file)
			# 判斷是否為某圖片集資料夾,是則讀取裡面的圖片
			if(os.path.isdir(filedir)):
				self.kinds.append(file)
				self.patharray.append(filedir)
	'''
		讀取資料
	'''
	def readData(self):
		count = 0

		for path in self.patharray:
			piclist = os.listdir(path)
			self.piccount.append(len(piclist))
			print('piccount:',self.piccount)
			for pic in piclist:
				picdir = os.path.join(path,pic)
				print('picdir:',picdir)
				picarray= cv2.imread(picdir,cv2.IMREAD_GRAYSCALE)
				picarray = cv2.resize(picarray,(32,32),interpolation=cv2.INTER_CUBIC)
				#picarray = picarray.flatten()
				print('picarray:',np.array(picarray).shape)
				self.picarray.append(picarray)
			print('selfpic:',np.array(self.picarray).shape)

		for i in self.piccount:
			for j in range(i):
				self.labels.append(count)
			count+=1
		print('labels',self.labels)

	'''
		改變陣列型態
	'''
	def changetype(self,types='nparray'):
		if types=='nparray':
			nparray = np.array(self.picarray)
			return nparray

	'''
		標準化流程
	'''
	def normalization(self,types='MINMAX'):
		if types=='MINMAX':
			big = 0
			small = 255
			length = len(self.picarray)
			print('length:',length)
			array = []
			for i in range(length):
				array.append([])

			for num in self.picarray:
				for row in num:
					for col in row:
						if col>big:
							big = col
						if col<small:
							small = col
			print('big:',big,',small:',small)
			count = 0
			rowcount = 0
			colcount = 0
			for num in self.picarray:
				for row in num:
					for col in row:
						self.picarray[count][rowcount][colcount]=(col-small)/(big-small)
						colcount+=1
					colcount=0
					rowcount+=1
				rowcount=0
				count+=1	
			#print('arraylen:',np.array(array).shape)
			#self.picarray = array
			print('picarraynew:',np.array(self.picarray).shape)


	'''
		one-hot encoding
	'''
	def one_hot(self):
		x = np.array(self.labels)
		#print('eye:',np.eye(len(self.piccount))[x])
		self.labels = np.eye(len(self.piccount))[x]
		print('label_eye:',self.labels)


	'''
		設定dict
		20181127 問題:處理打包成json文件檔案 20181128 已解決
	'''
	def setDictionary(self):
		count = 0
		index = 0
		#print(len(self.dictionary))	
		for i in self.piccount:
			self.dictionary.append([])
			for number in range(i):
				self.dictionary[count].append({"filename":self.kinds[count],"datas":self.picarray[index].tolist(),"labels":self.labels[index]})
				index+=1
			count+=1

	def jsonpacking(self):
		for i in range(len(self.piccount)):
			print('jsonpack:',len(self.dictionary[i]))
			with open(self.connpath+"\\datas_%s.json"%self.kinds[i],'w') as file:
				json.dump(self.dictionary[i],file)
	'''
		載入json文件資料
	'''
	def loadjson(self):
		for file in os.listdir(self.connpath):
			if os.path.isfile(os.path.join(self.connpath,file)):
				with open(os.path.join(self.connpath,file),'r') as f:  
					batch =json.load(f)
					#print('batch:',batch[1]['filename'])

	def createmodel(self,model):
		model.add(Conv2D(filters=2,kernel_size=(3,3),padding='same',
input_shape=(32,32,1),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(filters=3,kernel_size=(3,3),padding='same',activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(7,activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(len(self.kinds),activation='softmax'))
		return model  

	def getsummary(self,model):
		return model.summary()

	def train(self,model,Xtrain,Ytrain):
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.fit(x=Xtrain,y=Ytrain,validation_split=0.0,epochs=300,batch_size=2,verbose=2)
		return model

	def getevaluate(self,model,Xtrain,Ytrain):
		return model.evaluate(Xtrain,Ytrain)

	def getpredict(self,model,path):
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
		npimg = np.array(img)
		npimg = np.expand_dims(npimg,axis=2)
		npimg = np.expand_dims(npimg,axis=0)
		print('npimg:',npimg.shape)
		predicts = model.predict(npimg)
		#print('predict',predicts)
		return predicts.tolist()

	def answer(self,predictlist):
		print(np.array(predictlist))
		biggest = -1;
		for select in range(len(self.kinds)):
			if predictlist[0][select] > biggest:
				print('rate:',predictlist[0][select])
				biggest = select
		print('select:',biggest)
		print("this is a %s"%self.kinds[biggest])



	'''
		取得dict
	'''
	def getDictionary(self):
		return self.dictionary

	'''
		用以取得各種路徑值
	'''
	def getpath(self,types='set'):
		if types == 'all':
			return self.allpath
		elif types=='set':
			return self.setpath
		elif types=='conn':
			return self.connpath
		elif types=='array':
			return self.patharray
	
	def getarray(self):
		return self.picarray
				
	def getlabels(self):
		return self.labels

