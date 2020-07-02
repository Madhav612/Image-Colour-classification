import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import norm
from PIL import Image

Im = Image.open('tiger.jfif','r')
Im.show()
pix_val = list(Im.getdata())
pix_val = np.array(pix_val)    #306585*3

k=4

k_mean = []
for i in range(100,20*k+100,20):
    k_mean.append([i,i,i])
k_mean = np.array(k_mean)

def assignment(data,k_mean):
    r = np.zeros((1,data.shape[0]),dtype=int)
    l2 = [0]*k_mean.shape[0]
    for i in range(data.shape[0]):
        for k in range(k_mean.shape[0]):
            l2[k] = norm(data[i]-k_mean[k],2)
        min_ind = np.argmin(l2)    
        r[0][i] = min_ind    
    return r

def maximise(r,data,k):
    mean = np.zeros((k,3))
    sum_data = np.zeros((k,3))
    count = [1]*mean.shape[0]
    for i in range(data.shape[0]):
        sum_data[r[0][i]][0]+=data[i][0]
        sum_data[r[0][i]][1]+=data[i][1]
        sum_data[r[0][i]][2]+=data[i][2]
        count[r[0][i]] = count[r[0][i]]+1
    for i in range(mean.shape[0]):
        mean[i] = sum_data[i]/count[i]
        
    return mean

def diff(k_mean,prev_k_mean,eps):
    dis = k_mean-prev_k_mean
    for i in range(dis.shape[0]):
        d = norm(dis[i],2)
        print(d)
        if d > eps:
            return False
        else:
            return True

eps = 5

while True:
    #ASSIGNMENT STEP
    r = assignment(pix_val,k_mean)
    #MAXIMISATION STEP
    prev_k_mean = k_mean
    k_mean = maximise(r,pix_val,k)
    if diff(k_mean,prev_k_mean,eps):
        break

width, height = Im.size
r = np.reshape(r,(height,width))
k_mean = k_mean.astype(int)

type(k_mean)

for y in range(height):
    for x in range(width):
        Im.putpixel( (x, y), tuple(k_mean[r[y][x]]))    # putpixel provides read and write access to PIL.Image data 
Im.show()                                               # at a pixel level.
Im.save("hello.jpg")
