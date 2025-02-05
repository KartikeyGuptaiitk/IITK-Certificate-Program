import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, y = make_blobs(n_samples=2500,centers=4, n_features=2,random_state = 10)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=None, cmap='jet',s=10)
plt.suptitle('Original Data')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

y_pred=KMeans(n_clusters=4,random_state=0).fit_predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='jet',s=10)
plt.suptitle('K-Means Clusters')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

#Sum squared error
SSE = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)
    
plt.plot(range(1, 11), SSE)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.suptitle('SSE Plot for Elbow Method')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()


#Silhouette score
SS = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit_predict(X)
    score = silhouette_score(X, km.labels_, metric='euclidean')
    SS.append(score)
    
plt.plot(range(2, 11), SS)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette score")
plt.suptitle('Silhouette Score Plot for Kmeans')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()


