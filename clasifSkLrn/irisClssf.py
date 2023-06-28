from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from six import StringIO
import pydotplus

iris = load_iris()
# caracteristicas y especies
print(iris.feature_names)
features = iris.data
print(iris.target_names)
labels = iris.target
# print(iris.values()) < con esta linea imprimes todos los valores de los datos
# print(iris.data[0])

# Hacemos el clasificador
algoritmo = tree.DecisionTreeClassifier().fit(features, labels)

# prediction of new data
newDt = [[6.4, 3.1, 4.4, 1.2]]
print(algoritmo.predict(newDt))

# -----------------------------------------------------------------------
X = features
Y = labels
# Ahora debemos separar nuestros datos: split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.65)

# Entrenar el clasificador
clf = tree.DecisionTreeClassifier().fit(X_train, Y_train)

# Predicciones para los datos test
prdct = clf.predict(X_test)
print('Accuracy Score: ', accuracy_score(Y_test, prdct))

# newdata
newData = [[6.4, 3.1, 4.4, 1.2]]
print(clf.predict(newData))

# -------------------- Data - Visualization --------------------------------
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names,
                     class_names=iris.target_names, filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
