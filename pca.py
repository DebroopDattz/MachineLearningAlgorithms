import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X,y=digits.data, digits.target
print(X.shape)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca.shape)
print(pca.explained_variance_ratio_)

#SP1
plt.scatter(X_pca[:,0],X_pca[:,1],c=y,cmap='tab10',s=30,alpha=0.7)
plt.show()
pca_full=PCA().fit(X)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_),marker='o')
plt.show()

