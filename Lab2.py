import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis


df = pd.read_csv('C:/Users/feavl/OneDrive/Рабочий стол/МТУСИ/3 курс/Управление данными/лабы/lab2/glass.csv')

var_names = list(df.columns) #получение имен признаков
labels = df.to_numpy('int')[:,-1] #метки классов
data = df.to_numpy('float')[:,:-1] #описательные признаки

data = preprocessing.minmax_scale(data)

# plt.rcParams["figure.figsize"] = (17,7)
#
# fig, axs = plt.subplots(2,4)
# scatter3 = axs[0,0].scatter(data[:,0],data[:,(1)],c=labels,cmap='hsv')
# legend3 = axs[0,0].legend(*scatter3.legend_elements())
# for i in range(data.shape[1]-1):
#     axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)],c=labels,cmap='hsv')
#     axs[i // 4, i % 4].set_xlabel(var_names[i])
#     axs[i // 4, i % 4].set_ylabel(var_names[i+1])
#     axs[i // 4, i % 4].legend(*scatter3.legend_elements())
# plt.show()

# pca = PCA(n_components = 4)
# pca_data = pca.fit(data).transform(data)
# exvarratio = pca.explained_variance_ratio_
# expvarsum = np.sum(exvarratio)
# print(expvarsum)

# fig, axs = plt.subplots(1,1)
# plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv')
# plt.show()

# pca_inverse_transform = pca.inverse_transform(pca_data)
#
# fig, axs = plt.subplots(2,4)
#
# for i in range(data.shape[1]-1):
#     axs[i // 4, i % 4].scatter(pca_inverse_transform[:,i],pca_inverse_transform[:,(i+1)],c=labels,cmap='hsv')
#     axs[i // 4, i % 4].set_xlabel(var_names[i])
#     axs[i // 4, i % 4].set_ylabel(var_names[i+1])
# plt.show()

# pca = PCA(n_components=2, svd_solver='arpack')
#
# pca_data = pca.fit(data).transform(data)
#
# fig, axs = plt.subplots(1, 1)
#
# axs.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
#
# plt.show()

# pca = KernelPCA(n_components=2, kernel='rbf')
# pca_data = pca.fit(data).transform(data)
# fig, axs = plt.subplots(1, 1)
# axs.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
#
# plt.show()

# pca = SparsePCA(n_components=2, alpha=1)
# pca_data = pca.fit(data).transform(data)
# fig, axs = plt.subplots(1, 1)
# axs.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
#
# plt.show()

pca = FactorAnalysis(n_components=2)
pca_data = pca.fit(data).transform(data)
fig, axs = plt.subplots(1, 1)
axs.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')

plt.show()
