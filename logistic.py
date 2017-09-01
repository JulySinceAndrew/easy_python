import numpy as np
from PIL import Image
import os

def open_data(str1,str0,n_size):
    imgdir=os.listdir(str1)
    X=[]
    Y=[]
    i=0
    for s in imgdir:
        with Image.open(str1+'/'+s) as img:
            img=img.resize((n_size,n_size))
            img=np.array(img)
            img=img.reshape(1,n_size*n_size*3)
            X.append(img[0])
            Y.append(1)
            print(i)
            i=i+1
    imgdir=os.listdir(str0)
    for s in imgdir:
        with Image.open(str0+'/'+s) as img:
            img=img.resize((n_size,n_size))
            img=np.array(img)
            img=img.reshape(1,n_size*n_size*3)
            X.append(img[0])
            Y.append(0)
            print(i)
            i=i+1
    X=np.array(X)
    X=X.T
    X=X.reshape(n_size*n_size*3,i)
    m=len(Y)
    Y=np.array(Y)
    Y=Y.reshape(1,m)
    return X,Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def init_zeros(n_size):
    w=np.zeros((n_size*n_size*3,1))
    w=w.reshape(n_size*n_size*3,1)
    b=0
    return w,b

def propagate(w, b, X, Y):
    argus=dict()
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    J=-(np.dot(Y,np.log(A.T)+np.dot(1-Y,np.log(1-A.T))))/m
    dw=np.dot(X,(A-Y).T)/m
    db=(A-Y).sum()/m
    argus['dw']=dw
    argus['db']=db
    argus['J']=J[0][0]
    return argus

def optimize(w, b, X, Y, num_iterations,n_size,learning_rate, print_cost = False):
    for i in range(num_iterations):
        argus=propagate(w,b,X,Y)
        w=w-learning_rate*argus['dw']
        b=b-learning_rate*argus['db']
        if print_cost and i%100==0:
            print('train for ',i,'times,cost is ',argus['J'],w,b)
    cost=argus['J']
    argus=dict()
    argus['w']=w
    argus['b']=b
    argus['cost']=cost
    return argus

def predict(w,b,X,Y):
    Z=np.dot(w.T,X)+b
    A=sigmoid(np.dot(w.T,X)+b)
    m=X.shape[1]
    Y_predict=np.zeros((1,m))
    for i in range(m):
        if A[0][i]>=0.5:
            Y_predict[0][i]=1
    return Y_predict

def predict_signal(w,b,s,n_size):
    with Image.open(s) as img:
        img=img.resize((n_size,n_size))
        #img.show()
        img=np.array(img)
        x=img.reshape(n_size*n_size*3,1)
        a=sigmoid(np.dot(w.T,x)+b)
        if a[0][0]>=0.5:
            print(a,' 这是一只猫')
        else:
            print(a,'这不是一只猫')

def logistic(X_train,Y_train,X_test,Y_test,n_size,num_iterations=2000,learning_rate=0.5,print_cost=False):
    w,b=init_zeros(n_size)
    argus=optimize(w,b,X_train,Y_train,num_iterations,n_size,learning_rate,print_cost)
    w=argus['w']
    b=argus['b']
    print('train_predict_start')
    Y_train_predict=predict(w,b,X_train,Y_train)
    print('train_predict_end\ntest_predict_start')
    Y_test_predict=predict(w,b,X_test,Y_test)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))
    return argus
def main():
    X_train,Y_train=open_data('label_img1','label_img0',60)
    X_test,Y_test=open_data('test_img1','test_img0',60)
    argus=logistic(X_train,Y_train,X_test,Y_test,60,num_iterations=900,learning_rate=0.005,print_cost=True)
    w=argus['w']
    b=argus['b']
    while True:
        s=input('shuru')
        try:
            predict_signal(w,b,s,60)
        except ValueError:
            pass

main()




