# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:23:53 2020

@author: Navya
"""


import matplotlib.pyplot as plt
import cv2
import numpy as np
import os,time
import re
import glob, os.path,ntpath

#white range
lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])


#listing down all the file names
frames = os.listdir('fframes/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

#reading frames
images=[]
for i in frames:
    img = cv2.imread('fframes/'+i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(25,25),0)
    images.append(img)

images=np.array(images)

nonzero=[]
for i in range((len(images)-1)):
    
    mask = cv2.absdiff(images[i],images[i+1])
    _ , mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    num = np.count_nonzero((mask.ravel()))
    nonzero.append(num)
#    print("IMAGES SHAPE",images[i].shape)  
#    plt.imshow(mask)
#    plt.show()
#    time.sleep(secs)
    
#exit()
x = np.arange(0,len(images)-1)
y = nonzero

#plt.figure(figsize=(20,4))
#plt.scatter(x,y)

threshold = 15 * 10e3
#for i in range(len(images)-1):
#    if(nonzero[i]>threshold): 
#        scene_change_idx = i
#        break
        
#frames = frames[:(scene_change_idx+1)]
#frames = frames[:167]
#exit
img= cv2.imread('fframes/' + frames[73])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(25,25),0)
plt.imshow(img,cmap='gray')

#plt.figure(figsize=(5,10))
#plt.imshow(gray,cmap='gray')

#_ , mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

#plt.figure(figsize=(5,5))
#plt.imshow(mask,cmap='gray')
#Next
#image, contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#img_copy = np.copy(gray)
#cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)
#plt.imshow(img_copy, cmap='gray')
#exit()
#num=8
cnt=0

for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    
    numer=min([w,h])
    denom=max([w,h])
    ratio=numer/denom
    

    if(x>=num and y>=num):
        xmin, ymin= x-num, y-num
        xmax, ymax= x+w+num, y+h+num
    else:
        xmin, ymin=x, y
        xmax, ymax=x+w, y+h

    if(ratio>=0.5 and ((w<=30) and (h<=40)) ):    
        print(cnt,x,y,w,h,ratio)
        cv2.imwrite("fpatch/"+str(cnt)+".png",img[ymin:ymax,xmin:xmax])
        img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        cnt=cnt+1
    
import os
import cv2
import numpy as np
import pandas as pd
folders=os.listdir('data/')

images=[]
labels= []
for folder in folders:
    files=os.listdir('data/'+folder)
    for file in files:
        img=cv2.imread('data/'+folder+'/'+file,0)
        img=cv2.resize(img,(25,25))
        
        images.append(img)
        labels.append(int(folder))

images = np.array(images)
features = images.reshape(len(images),-1)

from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val = train_test_split(features,labels, test_size=0.2, stratify=labels,random_state=0)

#Build the motion model
from sklearn.ensemble import RandomForestClassifier 
seed=0
#rfc = RandomForestClassifier(max_depth=3,random_state=seed) 
rfc = RandomForestClassifier(max_depth=3) 
rfc.fit(x_tr,y_tr)

from sklearn.metrics import classification_report
y_pred = rfc.predict(x_val)
print(classification_report(y_val,y_pred))
exit()
ball_df = pd.DataFrame(columns=['frame','x','y','w','h'])

#for idx in range(55,len(frames)):
#for idx in range(170,171):
for idx in range(200,250):
#    time.sleep(10)
    
#    img= cv2.imread('fframes/' + frames[idx])
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    gray = cv2.GaussianBlur(gray,(25, 25),0)
#    _ , mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
#    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img= cv2.imread('fframes/' + frames[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#    image, contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#    !rm -r patch/*
    filelist = glob.glob(os.path.join(r"./fpatch", "*.png"))
    for f in filelist:
        os.remove(f)
#    filelist = glob.glob(os.path.join(r"./fball", "*.png"))
#    for f in filelist:
#        os.remove(f)
#    print(idx,frames[idx],len(contours) ) 
    num=5
    cnt=0
    df = pd.DataFrame(columns=['frame','x','y','w','h'])
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])

        numer=min([w,h])
        denom=max([w,h])
        ratio=numer/denom

        if(x>=num and y>=num):
            xmin, ymin= x-num, y-num
            xmax, ymax= x+w+num, y+h+num
        else:
            xmin, ymin= x,y
            xmax, ymax= x+w+w, y+h+h

        if(ratio>=0.):    
#            print(cnt,x,y,w,h,ratio)
            print ("IDX=",idx)
            df.loc[cnt,'frame'] = frames[idx]
            df.loc[cnt,'x']=x
            df.loc[cnt,'y']=y
            df.loc[cnt,'w']=w
            df.loc[cnt,'h']=h
            
            cv2.imwrite("fpatch/"+str(cnt)+".png",img[ymin:ymax,xmin:xmax])
#            time.sleep(10)
            cnt=cnt+1
    
    
    files=os.listdir('fpatch/')    
    if(len(files)>0):
    
        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        test=[]
        selected_files=[]
        img1=[]
        ffiles=[]
        for fyle in files:
            if os.stat(r"./fpatch/"+fyle).st_size > 100 :
                ffiles.append(ntpath.basename(fyle))
    
        for file in ffiles:
            img=cv2.imread('fpatch/'+file,0)
            img=cv2.resize(img,(25,25))
            test.append(img)                       
        test = np.array(test)

        test = test.reshape(len(test),-1)
        y_pred = rfc.predict(test)
        prob=rfc.predict_proba(test)

        if 0 in y_pred:
            ind = np.where(y_pred==0)[0]
            proba = prob[:,0]
            confidence = proba[ind]
            confidence = [i for i in confidence if i>0.5]
            if(len(confidence)>0):

                maximum = max(confidence)
                ball_file=files[list(proba).index(maximum)]

                img= cv2.imread('fpatch/'+ball_file)
#                time.sleep(10)
#                cv2.imwrite('fball/'+str(frames[idx]),img)
                cv2.imwrite('fball/'+str(frames[idx]),img)
#                time.sleep(10)

                no = int(ball_file.split(".")[0])
                ball_df.loc[idx]= df.loc[no]
            else:
                ball_df.loc[idx,'frame']=frames[idx]

        else:
            ball_df.loc[idx,'frame']=frames[idx]

print("Y_PRED LOCATION",np.where(y_pred==0)      )    
ball_df.dropna(inplace=True)
print(ball_df)  



files = ball_df['frame'].values

num=8
exit()
pvideo= "12.mp4"
#if pvideo=="12.mp4" :
n=55
for idx in range(len(ball_df)):    
    #draw contours 
    img = cv2.imread('fframes/'+files[idx])
#    print("IDX=",idx)
    n=int(files[0][0:-4])
    x=ball_df.loc[n,'x']
    y=ball_df.loc[n,'y']
    w=ball_df.loc[n,'w']
    h=ball_df.loc[n,'h']
#    print(idx,x,y,w,h)
    xmin=x-num
    ymin=y-num
    xmax=x+w+num
    ymax=y+h+num
    print("RECTANGLE",idx,files[idx],(xmin, ymin), (xmax, ymax))
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    cv2.imwrite("fframes/"+files[idx],img)   
    
frames = os.listdir('fframes/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))
exit()
frame_array=[]
#breakpoint
for i in range(len(frames)):
    #reading each files
    img = cv2.imread('fframes/'+frames[i])
    height, width, layers = img.shape
    size = (width,height)
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter('frackball.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()


