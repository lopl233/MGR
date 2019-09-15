from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle
import time
import warnings

warnings.filterwarnings("ignore")


def analaze_dataset(trainingData, trainingLabels, testData, testLabels, classifier,
                    printDetails=False, enable_clustering=False, enable_pca=True, class_dim_test=[], class_dim_train=[]):

    failed = 0
    passed = 0

    if enable_pca:
        pca = PCA(n_components=80)
        trainingData = pca.fit_transform(trainingData)
        testData = pca.transform(testData)

    if enable_clustering:
        clusters_number = 3
        kmeans = KMeans(n_clusters=clusters_number, random_state=0).fit(class_dim_train)
        test_klusters = kmeans.predict(class_dim_test)
        traing_klusters = kmeans.predict(class_dim_train)

        start = time.time_ns()

        for x in range(0, clusters_number):
            k_test_labels = [testLabels[y] for y in range(0, len(testLabels)) if test_klusters[y] == x]
            k_test_data = [testData[y] for y in range(0, len(testData)) if test_klusters[y] == x]
            k_training_labels = [trainingLabels[y] for y in range(0, len(trainingLabels)) if traing_klusters[y] == x]
            k_training_data = [trainingData[y] for y in range(0, len(trainingData)) if
                               traing_klusters[y] == x]

            if len(k_test_data) > 0:
                if len(list(set(k_training_labels))) > 1:
                    classifier.fit(k_training_data, k_training_labels)
                    predicted = classifier.predict(k_test_data)
                else:
                    predicted = [k_training_labels[0] for f in k_test_labels]
                for y in range(len(predicted)):
                    if predicted[y] == k_test_labels[x]:
                        passed = passed + 1
                    else:
                        failed = failed + 1
                    if printDetails:
                        print(predicted[y], "|", testLabels[y], "|", predicted[y] == testLabels[y])
    else:
        start = time.time_ns()

        classifier.fit(trainingData, trainingLabels)
        predicted = classifier.predict(testData)
        for x in range(0, len(predicted)):
            if predicted[x] == testLabels[x]:
                passed = passed + 1
            else:
                failed = failed + 1
            if printDetails:
                print(predicted[x], "|", testLabels[x], "|", predicted[x] == testLabels[x])

    end = time.time_ns()
    return passed/(passed+failed)*100, abs(start-end)*100

#read data

testData = np.loadtxt('testData')
trainingData = np.loadtxt('trainingData')

with open ('labelsTraining', 'rb') as fp:
    labelsTraining = pickle.load(fp)

with open ('labelsTest', 'rb') as fp:
    labelsTest = pickle.load(fp)

with open ('testClassicDims', 'rb') as fp:
    testClassicDims = pickle.load(fp)

with open ('trainingClassicDims', 'rb') as fp:
    trainingClassicDims = pickle.load(fp)


#create classifiers
classifiers = [
    MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
]

classifiers2 = [
    MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=200,activation = 'relu',solver='adam',random_state=1),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=3),
    GaussianNB(),
]


print(len(testData[0]))
#Eksperyment 0 - Test sprawdności
""" 
perc, time1 = analaze_dataset(trainingData, labelsTraining, testData, labelsTest, classifiers[2], printDetails=True)
print(perc)
"""

#Eksperyment 1 - Test klastrowania
""" 
for x in range(0, len(classifiers)-1):
    perc, time1 = analaze_dataset(trainingData, labelsTraining, testData, labelsTest, classifiers[x])
    perc2, time2 = analaze_dataset(trainingData, labelsTraining, testData, labelsTest, classifiers2[x], enable_clustering = True, class_dim_test= testClassicDims, class_dim_train = trainingClassicDims)
    print(perc, "|", perc2, "|", time1, "|", time2, "|", (time2/(time1+1))*100)
"""

#Eksperyment 2 - Test PCA
""" 
for x in range(0, len(classifiers)-1):
    perc, time1 = analaze_dataset(trainingData, labelsTraining, testData, labelsTest, classifiers[x],enable_pca= False)
    perc2, time2 = analaze_dataset(trainingData, labelsTraining, testData, labelsTest, classifiers2[x])
    print(perc, "|", perc2, "|", time1, "|", time2, "|", (time2/(time1+1))*100)
"""