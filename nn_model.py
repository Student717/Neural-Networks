
# coding: utf-8

# In[3]:


import os
import os.path
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,TimeDistributed,LSTM
from keras import callbacks

#from subprocess import call


# In[4]:


images =[]
label_folders=[]
video_labels=[]
folder_array=[]
each_video=[]
PATH= 'frames2/'
frame_array=[]
for filename in os.listdir(PATH):
    if list(filename)[0] == '.':
        continue    
    newpath = PATH + filename + '/'
    video_array=[]
    label_folders2=[]
    
    count = 0
    
    y=0
    
    no_val_images=len(os.listdir(newpath))-1
    for imagefile in os.listdir(newpath):
        label_name=imagefile.split('_')[0]
        if  os.path.exists(newpath+imagefile):
            if imagefile.endswith(".jpeg") & ~imagefile.startswith("."):
                #print(imagefile)
                #label_folders.append(imagefile.split('_')[0])
                if not(count==no_val_images):
                    img = cv2.imread(newpath+imagefile)
                    #print(type(img))
                    #img = img/255.
                    #img /= 255.
                    #img = cv2.resize((img, (128,128)))
                    #print(img)   
                    
                    if(count<24):
                        images.append(img)
                        video_array.append(img)
                        count+=1
                    else:
                        break
    while(count<24):
         # files.append((os.path.join(newpath,imagefile),label)
                #random.shuffle(files)
        #label_folders.append(label_name)
        images.append(np.zeros((128,128,3)))
        video_array.append(np.zeros((128,128,3)))
        count+=1
    while (y<24):
        label_folders2.append(label_name)
        label_folders.append(label_name)
        y+=1
    each_video.append(label_name)
    video_labels.append(label_folders2)
    #print(label_folders)
    #print(len(images))
    folder_array.append(video_array)          
                #for i in range(len(files)):
    #print(label_folders)
    #print(newpath)
    #print(images)   
    #print(images.shape)
    #print(count)
    
video_labels=np.array(video_labels)
each_video=np.array(each_video)
print(video_labels.shape)
print(each_video.shape)
#print(each_video,file=open('each_vid.csv','a'))               
           # labels.append(files[i][1])


# In[5]:


print(len(label_folders))
print(len(images))
print(np.array(video_array).shape)
print(each_video)
print(np.array(folder_array).shape)
print(np.array(images).shape)


# In[6]:


folder_array2=[]

for vid in folder_array:
    video_array2=[]
    vid_len=len(vid)
    for img in vid :
        img = img/255.
        #print(img.shape)
        img = cv2.resize(img, (128,128))
        #print(img)
        #print(i)
        #print(video_array2)
        #print(img)
        #print(i_count)
        
        video_array2.append(img)
        #print(np.array(video_array2).shape)
        
    folder_array2.append(video_array2)
        
print(np.array(folder_array2).shape)


# In[7]:


print(np.array(folder_array2).shape)
print(len(label_folders))
print(np.array(video_array).shape)


# In[8]:


le = preprocessing.LabelEncoder()
le = le.fit(each_video)
print(le)
classes=le.classes_
print(classes)
y_label = le.transform(each_video)
print(len(le.classes_))
print(y_label)


# In[9]:


y_train = np.zeros((len(y_label),len(le.classes_)))

for i in range(len(y_label)):
    y_train[i][y_label[i]] = 1
    
print(y_train.shape)


y_train2 = np.zeros((len(le.classes_),len(y_label)))

for i in range(len(le.classes_)):
    y_train2[i][y_label[i]] = 1
    
print(y_train2.shape)


# In[10]:


X_train = np.array(folder_array2)
print(X_train.shape)


# In[11]:


from sklearn.cross_validation import train_test_split


# In[12]:


model = Sequential()

model.add(TimeDistributed(Conv2D(16, (2, 2),activation='relu'), input_shape=(24,128,128,3)))
model.add(TimeDistributed(Conv2D(16, (2,2), activation='relu')))
model.add(TimeDistributed(MaxPooling2D()))
                          
model.add(TimeDistributed(Conv2D(32, (2,2), activation='relu')))
model.add(TimeDistributed(Conv2D(32, (2,2),activation='relu')))
model.add(TimeDistributed(MaxPooling2D()))
                          
#model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(Dropout(0.25))

model.add(TimeDistributed(Conv2D(64, (2,2),activation='relu')))
model.add(TimeDistributed(Conv2D(64, (2,2),activation='relu')))
model.add(TimeDistributed(MaxPooling2D()))
        
#model.add(TimeDistributed(Conv2D(512, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(512, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Flatten()))

model.add(Dropout(0.45))
model.add(LSTM(32, return_sequences=False, dropout=0.4))

model.add(Dense(len(le.classes_), activation='softmax'))

model.summary()

#early_stopper=callbacks.EarlyStopping(patience=7)

#checkpointer = callbacks.ModelCheckpoint(filepath='model7.h5',monitor='val_loss',verbose=1,save_best_only=True)

#lrreducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)




# In[11]:


model = Sequential()

model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),activation='relu', padding='same'), input_shape=(24,128,128,3)))
model.add(TimeDistributed(Conv2D(32, (3,3),kernel_initializer="he_normal", activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(64, (3,3),padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(64, (3,3),padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(Dropout(0.25))

model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(128, (3,3),padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        
#model.add(TimeDistributed(Conv2D(512, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(512, (3,3),padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Flatten()))

model.add(Dropout(0.45))
model.add(LSTM(64, return_sequences=False, dropout=0.4))

model.add(Dense(len(le.classes_), activation='softmax'))

model.summary()


# In[13]:


early_stopper=callbacks.EarlyStopping(patience=4)

checkpointer = callbacks.ModelCheckpoint(filepath='modelNOW1.h5',monitor='val_loss',verbose=1,save_best_only=True)

lrreducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)


# In[14]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[147]:


#model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stopper])


# In[148]:



#X_train,X_test,y_train, y_test= train_test_split(X_train,y_train,test_size=10,random_state=50)


# In[14]:




#model.fit(X_train, y_train, batch_size = 32, epochs=50, shuffle=True,verbose=1,callbacks=[checkpointer,lrreducer,early_stopper], validation_data=(X_test, y_test))


# In[15]:


from sklearn import cross_validation


# In[16]:


cv = cross_validation.KFold(3,shuffle=False, random_state=None)

#int(np.random.uniform(30, 60, size=1))

for i in cv:
    X_train,X_test,y_train, y_test= train_test_split(X_train,y_train,test_size=10,random_state=40)
    model.fit(X_train, y_train, batch_size = 16, epochs=50, shuffle=True,verbose=1,callbacks=[checkpointer,lrreducer,early_stopper], validation_data=(X_test, y_test))

