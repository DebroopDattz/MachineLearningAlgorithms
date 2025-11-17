import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

iris=load_iris()
X,y=iris.data[:,[2,3]],iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=1234)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

kernels=['linear','poly','rbf']
models={}

for k in kernels:
    if(k=='poly'):
        clf=SVC(kernel=k,degree=3,gamma='auto')
    else:
        clf=SVC(kernel=k,gamma='auto')
    clf.fit(X_train,y_train)
    models[k]=clf
    y_pred=clf.predict(X_test)
    print(f"Kernel:{k.upper()}")
    print(f"Accuracy:{accuracy_score(y_test,y_pred)}")
    print(classification_report(y_test,y_pred,target_names=iris.target_names))
    