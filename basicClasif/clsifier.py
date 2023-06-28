from sklearn import tree

# Training data
labels = [0, 0, 1, 1, 0]
# features = legs, weight
features = [[0, 50], [0, 150000], [4, 5], [4, 6], [0, 0.05]]

algoritmo = tree.DecisionTreeClassifier().fit(features, labels)

# New data
newDt = [[0, 30]]
print(algoritmo.predict(newDt))
