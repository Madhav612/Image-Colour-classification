import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import norm


def read_file(file,A):
    f=open(file,"r")
    for line in f:
        num1,num2=line.split()
        A.append([float(num1),float(num2)])
    return A


# def mean(x):
#     return [np.sum(x[:,0])/x.shape[0],np.sum(x[:,1])/x.shape[0]]

mean = np.zeros((2,2))
mean[0] = [10,10]
mean[1] = [7,10]


def assignment(data,mean):
    r = np.zeros((1,data.shape[0]))
    print(r.shape)
    #return r
    for i in range(data.shape[0]):
        l2_0 = norm(data[i]-mean[0],2)
        l2_1 = norm(data[i]-mean[1],2)
        if l2_0 > l2_1:
            r[0][i] = 1
        else:
            r[0][i] = 0
    return r        
    
def find_mean(r,data):
    sum0 = np.zeros((1,2))
    sum1 = np.zeros((1,2))
    mean = np.zeros((2,2))
    count0 = 0
    count1 = 0
    for i in range(data.shape[0]):
        if r[0][i]==0:
            sum0=sum0+data[i]
            count0+=1
        else:
            sum1=sum1+data[i]
            count1+=1
    mean[0] = sum0/count0
    mean[1] = sum1/count1
    return mean

eps = 0.1    

def diff(mean,prev_mean,eps):
    dis = mean-prev_mean
    for i in range(dis.shape[0]):
        d = norm(dis[i],2)
        if d > eps:
            return False
        else:
            return True


def plot(data,r,c1,c2):
    dat_1=[]
    dat_2=[]
    for i in range(data.shape[0]):
        if(r[0][i]==0):
            dat_1.append([data[i][0] ,data[i][1]])
        else:
            dat_2.append([data[i][0],data[i][1]])
    dat_1=np.array(dat_1)
    dat_2=np.array(dat_2)
    dict_={1:'violet',2:'mediumspringgreen'}
    dict_2 = {1:'blue',2:'darkgreen'}
    patch1 = mpatches.Patch(color=dict_[c1], label='class{}'.format(c1))
    patch2 = mpatches.Patch(color=dict_[c2], label='class{}'.format(c2))
    fig , ax = plt.subplots()
    ax = plt.scatter(dat_1[:,0], dat_1[:,1], s=2, c=dict_[c1])
    ax = plt.scatter(dat_2[:,0], dat_2[:,1], s=2, c=dict_[c2])
    ax = plt.legend(handles=[patch2,patch1])
    #ax = plt.scatter(data1[:,0], data1[:,1], s=2, c=dict_2[c1])
    #ax = plt.scatter(data2[:,0], data2[:,1], s=2, c=dict_2[c2])
    plt.show()

if __name__ == "__main__":
    data1= []
    data2= []
    # data3= []
    data1=read_file("../data/Class2_1.txt", data1)
    data2=read_file("../data/Class2_2.txt", data2)
    # data3=read_file("../data/Class1_3.txt", data2)

    data = np.concatenate((data1,data2),axis=0)
    # data = np.concatenate((data,data3),axis=0)
    while True:
        #ASSIGNMENT STEP
        r = assignment(data,mean)
        #print(r)
        #MAXIMISATION STEP
        prev_mean = mean
        mean = find_mean(r,data)
        if diff(mean,prev_mean,eps):
            break
        print(mean)
    plot(data,r,1,2)
