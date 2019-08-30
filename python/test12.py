from sklearn.decomposition import PCA





pca = PCA(n_components=2)


x = [[1,2,4],[2,7,18],[1,1,1]]
y = [[1,1,1],[2,2,2],[3,3,3]]
principalComponents = pca.fit_transform(x)
principalComponents2 = pca.transform(y)
print(principalComponents)
print(principalComponents2)