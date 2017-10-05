import read_pose_json as tools
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import itertools


csv_dir = '/Users/thammingkeat/PycharmProjects/athlete_data.csv'
athlete_data = tools.read_from_csv(csv_dir)

#Set up data for training
# X is features while Y is target label
X = []
Y = []
for athlete in athlete_data:
    #Truncate to only 28 features for X, remaining features not required - eye and ear are removed
    X.append(map(float,athlete[0:27]))
    Y.append(map(int,athlete[36]))


#Split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.80)

#Convert list of lists --> list
Y_train = list(itertools.chain(*Y_train))
Y_test = list(itertools.chain(*Y_test))

#Create the multi layer perceptron
#Parameter details : http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

mlp = MLPClassifier(hidden_layer_sizes=(28,28), activation='logistic', max_iter=10000, verbose=True)
mlp.fit(X_train,Y_train)
predictions = mlp.predict(X_test)

print(classification_report(Y_test,predictions))



