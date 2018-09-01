import numpy as np
import pickle
import os
import h5py
import glob
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------------------
class LogisticRegression:
    def __init__(self):
        pass
    
    def init_params(self,dimension):
        w = np.zeros((dimension, 1))
        b = 0
        return w, b
    
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))
    
    def propagate(self, w, b, X, Y):
        m = X.shape[1]
        A = self.sigmoid(np.dot(w.T,X) + b)
        cost = (-1/m)*(np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))))
        dw = (1/m)*(np.dot(X, (A-Y).T))
        db = (1/m)*(np.sum(A-Y))
        cost = np.squeeze(cost)
        grads = {"dw": dw, "db": db}
        return grads, cost
    

    def optimize(self, w, b, X, Y, epochs, lr):
        costs = []
        for i in range(epochs):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - (lr*dw)
            b = b - (lr*db)
            if i % 100 == 0:
                costs.append(cost)
                print ("cost after %i epochs: %f" %(i, cost))
        params = {"w": w, "b": b}
        grads  = {"dw": dw, "db": db}
        return params, grads, costs
    
    def predict(self, X):
        m = X.shape[1]
        w = self.params["w"]
        b = self.params["b"] 
        Y_predict = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)

        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            if A[0, i] <= 0.5:
                Y_predict[0, i] = 0
            else:
                Y_predict[0,i]  = 1

        return Y_predict

    
    def fit(self, X_train, Y_train, epochs=2000, learning_rate=0.01):
        ini_w, ini_b = self.init_params(X_train.shape[0])
        self.params, self.grads, self.costs = self.optimize(ini_w, ini_b, X_train, Y_train, epochs, learning_rate)
   
        Y_predict_train = self.predict(X_train)
        print ("Accuracy of Training Dataset : {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))
        

    def predict_image(self,X):
        Y_predict = None
        w = self.params["w"]
        b = self.params["b"] 
        w = w.reshape(X.shape[0], 1)
        A = self.sigmoid(np.dot(w.T, X) + b)
        for i in range(A.shape[1]):
            if A[0, i] <= 0.5:
                Y_predict = 0
            else:
                Y_predict = 1
        return Y_predict


    def predict_my_images(self, test_img_paths, image_size, train_path):
        for test_img_path in test_img_paths:
            img_to_show    = cv2.imread(test_img_path, -1)
            img            = image.load_img(test_img_path, target_size=image_size)
            x              = image.img_to_array(img)
            x              = x.flatten()
            x              = np.expand_dims(x, axis=1)
            predict1        = self.predict_image(x)
            predict_label  = ""
            train_labels = os.listdir(train_path)
            if predict1 == 0:
                predict_label = str(train_labels[0])
            else:
                predict_label = str(train_labels[1])
            cv2.putText(img_to_show, predict_label, (30,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            plt.imshow(img_to_show)
            plt.show()            


#-----------------------------------------------------------------------------------------        
def save_model(model_obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model_obj, f)
    print("Model saved to {}".format(str(file_name)))    

#----------------------------------------------------------------------------------------- 
def open_saved_model(file_name):
    with open(file_name, 'rb') as f:
        clf = pickle.load(f)
    return clf    
 
#-----------------------------------------------------------------------------------------         
def prepare_image_dataset(train_path, test_path, image_size, num_train_images, num_test_images, num_channels=3):    
    train_labels = os.listdir(train_path)
    test_labels  = os.listdir(test_path)

    # image_size = (64,64)
    # num_train_images = 1498
    # num_test_images = 100
    # num_channels = 3 #RGB

    train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
    train_y = np.zeros((1, num_train_images))
    test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
    test_y  = np.zeros((1, num_test_images))

    #train dataset
    count = 0
    num_label = 0
    for i, label in enumerate(train_labels):
        cur_path = train_path + "/" + label
        for image_path in glob.glob(cur_path + "/*.jpg"):
            img = image.load_img(image_path, target_size=image_size)
            x   = image.img_to_array(img)
            x   = x.flatten()
            x   = np.expand_dims(x, axis=0)
            train_x[:,count] = x
            train_y[:,count] = num_label
            count += 1
        num_label += 1

    count = 0 
    num_label = 0 
    for i, label in enumerate(test_labels):
        cur_path = test_path + "/" + label
        for image_path in glob.glob(cur_path + "/*.jpg"):
            img = image.load_img(image_path, target_size=image_size)
            x   = image.img_to_array(img)
            x   = x.flatten()
            x   = np.expand_dims(x, axis=0)
            test_x[:,count] = x
            test_y[:,count] = num_label
            count += 1
        num_label += 1

    train_x = train_x/255.
    test_x  = test_x/255.

    print ("train_labels : " + str(train_labels))
    print ("train_x shape: " + str(train_x.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x shape : " + str(test_x.shape))
    print ("test_y shape : " + str(test_y.shape))
    
    return train_x, test_x, train_y, test_y

#----------------------------------------------------------------------------------------- 
def accuracy(y_pred, y_true):
    return 100-np.mean(np.abs(y_pred - y_true)) * 100

#----------------------------------------------------------------------------------------- 
