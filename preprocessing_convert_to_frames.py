
# coding: utf-8

# In[1]:


import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn import preprocessing
from subprocess import call


# In[14]:


PATH_1 = 'actions/'

files = []
images = []
labels = []


for filename in os.listdir(PATH_1):
    if list(filename)[0] == '.':
        continue
    newpath = PATH_1 + filename + '/'
    label = filename
    labels.append(label)
    print(label+"...")
    vidfile_count=1
    for vidfile in os.listdir(newpath):
        if vidfile.endswith(".avi"):
            if list(vidfile)[0] == '.'  :
                continue
           # print(os.getcwd())
            newpath_2 = 'frames2/'
            newpath_1= newpath_2+label+str(vidfile_count)
            
            try:
                if not os.path.exists(newpath_1):
                    os.makedirs(newpath_1)
            except OSError:
                print ("Error: Creating directory of %s",label+str(vidfile_count))
        
            vidcap = cv2.VideoCapture(newpath+vidfile)
            count = 0
            
            seconds=0.3
            fps=vidcap.get(cv2.CAP_PROP_FPS)
            print(fps)
            multiplier = fps*seconds
            
            success = True
            while success:
                
                frameId=int(round(vidcap.get(1)))
                success,image = vidcap.read()
                
                #frameno=1
                
                if frameId % multiplier == 0:
                    cv2.imwrite(newpath_1 +"/"+ label +"_frame%2d.jpeg" % (frameId/3), image)
                    #frameno+=1
                
                    #print (type(image))
                    #print(newpath_1+"/frame%d.jpeg" % count, image)
                    # save frame as JPEG file
                    #image = image/255.
                    #image = cv2.resize(image, (128,128))
                count += 1
            vidcap.release()
            cv2.destroyAllWindows()
            vidfile_count+=1
            #print(vidfile+'_'+str(count))
    

