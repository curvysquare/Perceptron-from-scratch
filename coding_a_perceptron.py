#Student ID: 201680340
import numpy as np
import pandas as pd
from numpy import random
random.seed(1)
# the "Dataset" class below takes the training and test data and stores
# them as class attributes in the pandas dataframe form.
class Dataset:
    def __init__(self, train_csv, test_csv):
        self.train_data = pd.read_csv(train_csv)
        self.test_data = pd.read_csv(test_csv)
    # The method below takes in ClassA (positive class), classB (negative class),dataframe, and
    # the boolean parameter OneVsRest (OVR) which if true, all other classes other than classA
    # become negative classes. It replacesclass labels with 1 and -1, and shuffles the data
    # The method returns two numpy arrays: dataset_X (features) and dataset_Y (labels).
    def binary_data_formatter(self, classA, classB, dataframe, OVR):
        df = dataframe.copy()
        df.columns = ['x1', 'x2', 'x3','x4', 'class']
        if OVR == False:
            df = df.loc[df['class'].isin([classA, classB])]
            df['class'] = df['class'].replace({classA: 1, classB:-1})
        if OVR == True:
            df.loc[df["class"] != classA, "class"] = -1
            df.loc[df["class"] == classA, "class"] = 1
        #line below is used to shuffle
        df = df.sample(frac=1)
        dataset_X = df.loc[:, 'x1':'x4']
        # An additional feature X0 is added with value 1 to include the bias (w0)in the dot product.
        dataset_X.insert(0, 'Always_on', 1 )
        dataset_Y =  df.loc[:, 'class']
        dataset_X = dataset_X.to_numpy()
        dataset_Y = dataset_Y.to_numpy()
        return dataset_X, dataset_Y
# an instance of the datset class is created below
dataset_binary = Dataset('train.data', 'test.data')
# for each class pair, the train and test datsets accsessed from the
# corresponding "dataset" object attributes and are formatted using the "binary_data_formatter"
# method above. Each variable therefore conaints a tuple of the object and label sets.
C1C2_train = dataset_binary.binary_data_formatter('class-1', 'class-2',dataset_binary.train_data, OVR=False)
C1C2_test = dataset_binary.binary_data_formatter('class-1', 'class-2',dataset_binary.test_data, OVR=False)
C2C3_train = dataset_binary.binary_data_formatter('class-2', 'class-3',

dataset_binary.train_data, OVR=False)
C2C3_test = dataset_binary.binary_data_formatter('class-2', 'class-3',
dataset_binary.test_data, OVR=False)
C1C3_train = dataset_binary.binary_data_formatter('class-1', 'class-3',
dataset_binary.train_data, OVR=False)
C1C3_test = dataset_binary.binary_data_formatter('class-1', 'class-3',
dataset_binary.test_data, OVR=False)
# the perceptron class below is intialised with the training set for it to be trained on.
# the training objects and labels are accessed from the training_set tuple created above.
class Perceptron:
    def __init__(self, training_set):
        self.training_set = training_set
        self.training_objects = training_set[0]
        self.training_labels = training_set[1]
        self.n_of_features = len(self.training_objects[1])
        self.n_of_objects = len(self.training_objects)
        # the weights are intialised randomly between -10 and 10
        # since the first element (W0) in the weight vector is the bias
        # this is als intialied randomly.
        self.weight_vector = np.random.randint(-10, 10, self.n_of_features)
        self.accuracy = 0
    # for binary classification, lamda coefficient of l2 regularization is equal to zero.
    def Train(self, max_iter, lam):
        for i in range(max_iter):
            for j in range(self.n_of_objects):
                X_vector = self.training_objects[j]
                Y = self.training_labels[j]
                activation_score = np.dot(X_vector, self.weight_vector.T)
                if activation_score * Y  <= 0:
                    #self.weight_vector[0] is the bias
                    self.weight_vector[0] = self.weight_vector[0] + Y
                    # the for loop below has range starting at 1, so the bias atndex 0 is excluded.
                    for k in range(1, self.n_of_features):
                        self.weight_vector[k] = self.weight_vector[k]*(1-(2*lam)) + (X_vector[k] *Y)

                else:
                    for l in range(1, self.n_of_features):
                        self.weight_vector[l] = self.weight_vector[l]* (1-(2*lam))
        return self.weight_vector
    
def Test(self, test_set,):
    # reset accuracy
    self.set_name = test_set
    self.accuracy = 0
    test_objects = test_set[0]
    test_labels = test_set[1]
    prediction_list = []

    for i in range(len(test_objects)):
        target_label = test_labels[i]
        activation_score = np.dot(test_objects[i],  self.weight_vector.T)
        prediction_list.append(int(1*np.sign(activation_score)))
        if activation_score * test_labels[i] <= 0:
            pass 
        else:
            self.accuracy += 1
    accuracy_percentage = (self.accuracy/len(test_labels))*100
    # code below reformates the list of predictions and test labels
    # for the use in construction a confusion matrix since the interger values
    # of each prediction and test label are used to incremenent the corresponding
    # segment of the confusion matrix.
    for i in range(len(prediction_list)):
        if prediction_list[i] == -1:
            prediction_list[i] = 0
            pass
    for i in range(len(test_labels)):
        if test_labels[i] == -1:
            test_labels[i] = 0
            pass
    prediction_list = np.array(prediction_list)
    test_labels = np.array(test_labels)
    confusion_matrix=np.zeros([2,2])
    for prediction, target in zip(prediction_list, test_labels):
        confusion_matrix[prediction][target]+= 1
    # print(confusion_matrix)
    return (f'{accuracy_percentage:.4}')
# each binary classifier perceptron is initialzed with its corresponding dataset.
# it is then trained using the "Train" perceptron class method
# it is then tested using the "Test" perceptron class method which gives accuracy as
# an output

perc_C1C2 = Perceptron(C1C2_train)
perc_C1C2.Train(max_iter=20, lam=0)
print(" C1C2 Train accuracy:", perc_C1C2.Test(C1C2_train))
print(" C1C2 Test accuracy:", perc_C1C2.Test(C1C2_test))
perc_C2C3 = Perceptron(C2C3_train)
perc_C2C3.Train(max_iter=20, lam=0)
print(" C2C3 Train accuracy:", perc_C2C3.Test(C2C3_train))
print(" C2C3 Test accuracy:", perc_C2C3.Test(C2C3_test))
perc_C1C3 = Perceptron(C1C3_train)

perc_C1C3.Train(max_iter=20, lam=0)
print(" C1C3 Train accuracy:", perc_C1C3.Test(C1C3_train))
print(" C1C3 Test accuracy:", perc_C1C3.Test(C1C3_test))
# The OVR_model class inherits the dataset and perceptron classes.
# it Initialised with a dataset instance, which is saved as a class attribute to self.datset.
# the "classifier_params" attribute is a dictionary that saves the parameters of each trained classifier
# there are two OVR class methods for testing. One that uses the training data set as the test set
# and the other that uses the test data as the test set. The accuracies are outputed from both.

class OVR_model(Dataset, Perceptron):
    def __init__(self, train_CSV, test_CSV):
        self.dataset_OVR = Dataset(train_CSV,test_CSV)
        # the two lines below are used to increase legibility
        self.train_data =self.dataset_OVR.train_data
        self.test_data =self.dataset_OVR.test_data
        self.train_data.columns = ['x1', 'x2', 'x3','x4', 'class']
        # a list containing all the unique class labels
        self.class_list = self.train_data['class'].unique()
        # the list of unique class labels are used as the keys for a dictionary
        self.classifier_params = dict.fromkeys(self.class_list, [])
        self.accuracy_test_set = 0
    # for each class label("classA"), the Dataset class method of binary_data_formatter is used
    # to format the training set to +1 for classA and -1 for all other classes since the Boolean
    # OVR parameter is set to true. The perceptron for each classifier is then intilized with this training
    # set and then subsequently trained.
    # the trained parameters (the bias and weights) of the weight vector are then saved to the "classifier_params"
    # dictionary.
    
    def Train_classifiers(self, max_iter, lam):
        for classA in self.class_list:
            CA_OVR_training_data = self.dataset_OVR.binary_data_formatter(classA,classB=0, dataframe=self.train_data,OVR =True )
            C = Perceptron(CA_OVR_training_data)
            C.Train(max_iter, lam)
            self.classifier_params[classA] = C.weight_vector
            
    def predict_accuracy_test_set(self):
        df = self.test_data
        df.columns = ['x1', 'x2', 'x3','x4', 'class']
        Test_X = df.loc[:, 'x1':'x4']
        Test_X.insert(0, 'Always_on', 1 )
        Test_Y =  df.loc[:, 'class']
        Test_X = Test_X.to_numpy()
        Test_Y = Test_Y.to_numpy()
        test_pred_lables_list = []
        for i in range(len(Test_X)):
            X = Test_X[i]
            class_list = list(self.classifier_params.keys())
            score_list = []
            for class_key in class_list:
                per_class_model_params = self.classifier_params[class_key]
                activation_score = np.dot(X, per_class_model_params.T)
                score_list.append(activation_score)
            vote = str(class_list[np.argmax(score_list)])
            test_pred_lables_list.append(vote)
            if vote == Test_Y[i]:
                self.accuracy_test_set += 1
        self.test_accuracy = self.accuracy_test_set/len(Test_X)*100
        # the remainder of the code in this function is to produce a confusion matrix for the
        # purpose of analysing the components that make up the accuracy for the testset.
        for i in range(len(test_pred_lables_list)):
            if test_pred_lables_list[i] == 'class-1':
                test_pred_lables_list[i] = 0
            if test_pred_lables_list[i] == 'class-2':
                test_pred_lables_list[i] = 1
            if test_pred_lables_list[i] == 'class-3':
                test_pred_lables_list[i] = 2
        for i in range(len(Test_Y)):
            if Test_Y[i] == 'class-1':
                Test_Y[i] = 0
            if Test_Y[i] == 'class-2':
                Test_Y[i] = 1
            if Test_Y[i] == 'class-3':
                Test_Y[i] = 2
        test_pred_lables_list = np.array(test_pred_lables_list)
        Test_Y = np.array(Test_Y)
        confusion_matrix=np.zeros([3,3])
        for prediction, target in zip(test_pred_lables_list, Test_Y):
            confusion_matrix[prediction][target]+= 1
        #print(confusion_matrix)
    def predict_accuracy_train_set(self):
        df = self.train_data
        df.columns = ['x1', 'x2', 'x3','x4', 'class']
        Test_X = df.loc[:, 'x1':'x4']
        Test_X.insert(0, 'Always_on', 1 )
        Test_Y =  df.loc[:, 'class']
        Test_X = Test_X.to_numpy()
        Test_Y = Test_Y.to_numpy()
        self.train_accuracy_count = 0
        self.train_pred_lables_list = []
        for i in range(len(Test_X)):
            X = Test_X[i]

            class_list = list(self.classifier_params.keys())
            score_list = []
            for class_key in class_list:
                per_class_model_params = self.classifier_params[class_key]
                activation_score = np.dot(X, per_class_model_params.T)
                score_list.append(activation_score)
            vote = str(class_list[np.argmax(score_list)])
            self.train_pred_lables_list.append(vote)
            if vote == Test_Y[i]:
                self.train_accuracy_count += 1
        self.train_accuracy = self.train_accuracy_count/len(Test_X)*100
        # the remainder of the code in this function is to produce a confustion matrix for the
        # purpose of analysing the components that make up the accuracy for the training set.
        for i in range(len(self.train_pred_lables_list)):
            if self.train_pred_lables_list[i] == 'class-1':
                self.train_pred_lables_list[i] = 0
            if self.train_pred_lables_list[i] == 'class-2':
                self.train_pred_lables_list[i] = 1
            if self.train_pred_lables_list[i] == 'class-3':
                self.train_pred_lables_list[i] = 2
        for i in range(len(Test_Y)):
            if Test_Y[i] == 'class-1':
                Test_Y[i] = 0
            if Test_Y[i] == 'class-2':
                Test_Y[i] = 1
            if Test_Y[i] == 'class-3':
                Test_Y[i] = 2
        self.train_pred_lables_list = np.array(self.train_pred_lables_list)
        Test_Y = np.array(Test_Y)
        confusion_matrix=np.zeros([3,3])
        for prediction, target in zip(self.train_pred_lables_list, Test_Y):
            confusion_matrix[prediction][target]+= 1
        #print(confusion_matrix)
        
# the "OVR_program" function below creates the "OVR" instance of the OVR_model class.
# As outlined above, this initialisation creates the "classifier_params" dictionary
# the "Train_classifiers" class method is used to intialise and train the three OVR classifiers
# the "predict_accuracy_test_set" and "predict_accuracy_train_set" class methods are used to
# output the corresponding accuracy.

def OVR_program(train_data, test_data, max_iter, lam):
    OVR = OVR_model(train_data, test_data)
    OVR.Train_classifiers(max_iter, lam)
    OVR.predict_accuracy_train_set()
    print("OVR train accuracy:", f'{OVR.train_accuracy:.4}')
    OVR.predict_accuracy_test_set()
    print("OVR test accuracy",f'{OVR.test_accuracy:.4}')

# Question 4
OVR = OVR_program('train.data:', 'test.data', max_iter=20 , lam = 0)
# Question 5
for coef in [0.01, 0.1, 1, 10, 100]:
    print("lamda coefficient value:", coef)
    OVR = OVR_program('train.data', 'test.data', max_iter=20 , lam = coef)