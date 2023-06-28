from sklearn import tree

# Training data
labels = [0, 0, 1, 0, 1, 1, 0, 0]
# features = wheels, seats
features = [[2, 2], [2, 2], [3, 4], [2, 2], [4, 4], [4, 4], [3, 2], [2, 2]]

algoritmo = tree.DecisionTreeClassifier().fit(features, labels)

# New data
newDt = [[2, 3], [3, 4]]
print(algoritmo.predict(newDt))
