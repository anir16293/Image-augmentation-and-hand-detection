import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Create a collection of images object for a dataset of 28x28 images
# Dataset assumed to have features in the following order: labels, pixel0,...,pixel783
class ImageCSV:
    def __init__(self,path):
        self.df=pd.read_csv(path)
        self.image_memo={}
    def get_image_i(self,i):
        if i not in self.image_memo:
            image=self.df.loc[i][1:].values.reshape(28,28)#Change image dimensions here in case of different image resolution
            self.image_memo[i]=image
        return(self.image_memo[i])
    def display_image_i(self,i):
        plt.imshow(self.get_image_i(i),cmap='gray')
        plt.show()
    def get_label_i(self,i):
        return(self.df.label.loc[i])
    def trainTestSplit(self,trainRatio=0.8):
        X=self.df.values[:,1:]
        y=self.df.values[:,0]
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=trainRatio)
        return((zip(X_train,y_train),zip(X_test,y_test)))
    def augment_image_i(self,i):
        imgVec=self.get_image_i(i)
        imgSize=imgVec.shape[0]*imgVec.shape[1]
        augmentationVec=np.random.binomial(1,0.3,10)
        output=self.randomAugmentation(imgVec,augmentationVec)
        return(output)
    def get_augmented_dataset(self,aug_per_img=10):
        m=self.df.shape[0]
        images=[]
        flatten=lambda list1:[v for sublist in list1 for v in sublist]
        for i in range(m):
            label1=self.get_label_i(i)
            for j in range(aug_per_img):
                img1=self.augment_image_i(i)
                img1=[label1]+flatten(img1)
                images.append(img1)
        return(np.array(images))
    def imageResize(self,imgVec,dimX=28,dimY=28):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        return(cv2.resize(imgVec,(dimX,dimY)))
    def scaleImage(self,imgVec,fx=0.5,fy=0.5):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        return(cv2.resize(imgVec,None,fx=fx,fy=fy))
    def translateImage(self,imgVec,xShift,yShift):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        rows,cols=imgVec.shape
        output=np.zeros((rows,cols),dtype=np.uint8)
        if xShift>0 and yShift>0:
            output[xShift:,yShift:]=imgVec[0:rows-xShift,0:cols-yShift]
        elif xShift<=0 and yShift>0:
            output[0:rows+xShift,yShift:]=imgVec[-xShift:,0:cols-yShift]
        elif xShift>0 and yShift<=0:
            output[xShift:,0:cols+yShift]=imgVec[0:rows-xShift,-yShift:]
        elif xShift<=0 and yShift<=0:
            output[0:rows+xShift,0:cols+yShift]=imgVec[-xShift:,-yShift:]
        return(output)
    def flipImage(self,imgVec):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        return(imgVec[:,::-1])
    def rotateImage(self,imgVec,theta):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        return(rotate(imgVec,theta))
    def saltImage(self,imgVec):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        return(random_noise(imgVec,mode='salt'))
    def lightImage(self,imgVec,lightingMean):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        return(random_noise(imgVec,mean=lightingMean))
    def contrastImage(self,imgVec,lowerBrightness=0,upperBrightness=255):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        return(rescale_intensity(imgVec,in_range=(lowerBrightness,upperBrightness)))
    def randomCrop(self,imgVec,dy,dx):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        y=np.random.randint(0,imgVec.shape[0]-dy)
        x=np.random.randint(0,imgVec.shape[1]-dx)
        return(imgVec[y:y+dy,x:x+dx])
    def shearImage(self,imgVec,shear=0.2):
        try:
            imgVec=cv2.cvtColor(imgVec,cv2.COLOR_RGB2GRAY)
        except:
            pass
        tf1=AffineTransform(shear=shear)
        return(warp(imgVec,inverse_map=tf1))
    def randomAugmentation(self,imgVec,augmentationVec):
        if augmentationVec[2]==1:
            xShift=get_truncated_normal(mean=0,sd=20,low=-imgVec.shape[0]/40,upp=imgVec.shape[0]/40).rvs()
            yShift=get_truncated_normal(mean=0,sd=20,low=-imgVec.shape[1]/40,upp=imgVec.shape[1]/40).rvs()
            imgVec=self.translateImage(imgVec, xShift=int(xShift), yShift=int(yShift))
        if augmentationVec[3]==1:
            imgVec=self.flipImage(imgVec)
        if augmentationVec[4]==1:
            theta1=get_truncated_normal(mean=0,sd=30, low=-180, upp=180).rvs()
            imgVec=self.rotateImage(imgVec,theta=theta1)
        if augmentationVec[5]==1:
            imgVec=self.saltImage(imgVec)
        if augmentationVec[6]==1:
            lightMean=get_truncated_normal(mean=0, sd=0.25, low=-0.25, upp=0.25).rvs()
            imgVec=self.lightImage(imgVec,lightingMean=0.1)
        if augmentationVec[7]==1 and augmentationVec[6]!=1:
            x1=get_truncated_normal(mean=100, sd=10, low=50, upp=150).rvs()
            imgVec=self.contrastImage(imgVec, lowerBrightness=x1, upperBrightness=x1+80)
        if augmentationVec[8]==1:
            imgVec=self.randomCrop(imgVec, dy=int(0.9*imgVec.shape[0]), dx=int(0.9*imgVec.shape[1]))
        if augmentationVec[9]==1:
            shearRatio=get_truncated_normal(mean=0, sd=0.5, low=-1, upp=1).rvs()
            imgVec=self.shearImage(imgVec, shear=shearRatio)
        return(imgVec)
