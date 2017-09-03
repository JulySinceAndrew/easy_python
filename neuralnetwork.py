import numpy as np
import os
from PIL import Image

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

def init(layers_dim):
    L=len(layers_dim)
    parameters=dict()
    for i in range(1,L):
        parameters['W'+str(i)]=np.random.randn(layers_dim[i],layers_dim[i-1])*0.01
        parameters['b'+str(i)]=np.zeros((layers_dim[i],1))
    return parameters

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.where(Z>0,Z,0)

def relu_back(Z):
    return np.where(Z>0,1,0)

def forward(X,parameters):
    L=len(parameters)//2
    cache=dict()
    A_prev=X
    cache['A0']=X
    for i in range(1,L):
        Z=np.dot(parameters['W'+str(i)],A_prev)+parameters['b'+str(i)]
        A_prev=relu(Z)
        cache['Z'+str(i)]=Z
        cache['A'+str(i)]=A_prev
    Z=np.dot(parameters['W'+str(L)],A_prev)+parameters['b'+str(L)]
    AL=sigmoid(Z)
    cache['Z'+str(L)]=Z
    cache['A'+str(L)]=AL
    return AL,cache

def J_cost(AL,Y):
    m=Y.shape[1]
    return -(np.dot(Y,np.log(AL.T))+np.dot(1-Y,np.log(1-AL.T)))/m

def back(parameters,cache,Y):
    L=len(parameters)//2
    dZ=cache['A'+str(L)]-Y
    grads=dict()
    m=Y.shape[1]
    for i in reversed(range(1,L+1)):
        grads['dW'+str(i)]=np.dot(dZ,cache['A'+str(i-1)].T)/m
        grads['db'+str(i)]=dZ.sum(axis=1)/m
        grads['db'+str(i)]=grads['db'+str(i)].reshape(dZ.shape[0],1)
        if i==1:
            break
        dZ=np.dot(parameters['W'+str(i)].T,dZ)*relu_back(cache['Z'+str(i-1)])
    return grads

def up_date(parameters,grads,learning_rate):
    L=len(parameters)//2
    for i in range(1,L+1):
        parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*grads['dW'+str(i)]
        parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*grads['db'+str(i)]
    return parameters

def L_neural_network(X,Y,layers_dim,learning_rate,num_iterations,wait_convergence=True,esplion=0.0001,print_cost=False):
    parameters=init(layers_dim)
    cost=[]
    for i in range(num_iterations):
        AL,cache=forward(X,parameters)
        if i%100==0:
            cost.append(J_cost(AL,Y))
            if print_cost:
                print('train for %d times,the cost is %f'%(i,cost[i//100]))
            if wait_convergence and i>0:
                num=i//100
                if abs(cost[num]-cost[num-1])<esplion:
                    print('The algorithm has convergenced!')
                    break
        grads=back(parameters,cache,Y)
        parameters=up_date(parameters,grads,learning_rate)

    return parameters

def predict_dataset(X,parameters):
    AL,cache=forward(X,parameters)
    Y_prediction=np.zeros((1,AL.shape[1]))
    for i in range(AL.shape[1]):
        if AL[0][i]>=0.5:
            Y_prediction[0][i]=1
    return Y_prediction

def predict_signalefile(s,parameters):
    try :
        img=Image.open(s)
        img = img.resize((60,60))
        img = np.array(img)
        X = img.reshape( 60 * 60 * 3,1)
    except FileNotFoundError:
        print('No such file!')
    AL,cache=forward(X,parameters)
    AL=AL[0][0]
    if AL>=0.5:
        print(AL,' 这是一只猫')
    else:
        print(AL,' 这不是一只猫')

def main():
    X_train,Y_train=open_data("label_img1","label_img0",60)
    X_test,Y_test=open_data("test_img1","test_img0",60)
    layers_dim=[60*60*3,20,7,5,1]
    parameters=L_neural_network(X_train, Y_train, layers_dim, learning_rate=0.005, num_iterations=3000, wait_convergence=False, esplion=0.0001, print_cost=True)
    Y_train_predict=predict_dataset(X_train,parameters)
    Y_test_predict=predict_dataset(X_test,parameters)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))
    while True:
        s=input('input')
        predict_signalefile(s,parameters)

main()