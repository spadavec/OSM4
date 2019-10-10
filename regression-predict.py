import csv
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import operator
import numpy
import argparse
from keras.models import load_model
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from sklearn.model_selection import train_test_split
import pandas
from rdkit import DataStructs
from itertools import combinations
import random
import glob
import keras
from sklearn.linear_model import LogisticRegression

class Regression(object):
    def __init__(self, smiles_train, smiles_test, ic50_train, classes_train, id_test):
        self.id_test = id_test 
        self.final_results = [] 
        # Split data into necessary groups
        self.train_potency = [math.log(float(x+0.0000001),10) for x in ic50_train]
        self.train_smiles = smiles_train
        self.train_fps = self.generate_fingerprints(smiles_train)
        self.classes_train = classes_train 

        self.test_smiles = smiles_test
        self.test_fps = self.generate_fingerprints(smiles_test)

        # Compile the Sequential Model1
        self.create_model()

        # Save the model so that people don't have to train
        self.save_sequential_model()

        # Train the model
        self.fit_model()

        # Make predictions
        self.potency_predictions = self.model_predict()
        
        self.create_lr_model()
        self.fit_lr_model()
        self.lr_model_predict()

        self.potency_predictions = [10**x[0] for x in self.potency_predictions]
        
        
        self.final_results.append([self.potency_predictions, self.class_preds])
        
        ensemble = []
        ic50_alone = []

        for i,x in enumerate(self.potency_predictions):
            if x < 200 and self.class_preds[i] == 1:
                ensemble.append('Yes')
            else:
                ensemble.append('No')
            if x < 300:
                ic50_alone.append('Yes')
            else:
                ic50_alone.append('No')
        
        out_df = pandas.DataFrame()
        out_df['ids'] = self.id_test
        out_df['IC50 (nM)'] = self.potency_predictions
        out_df['Class Active'] = self.class_preds 
        out_df['IC50 Alone'] = ic50_alone
        out_df['Ensemble'] = ensemble 
        out_df = out_df.sort_values(by=['IC50 (nM)'], ascending=True)
        out_df.to_csv('ITX_S4_Predictions.csv', index=False)
    def create_lr_model(self):
        self.lr_model = LogisticRegression(solver='liblinear')

        return self.model
    
    def fit_lr_model(self):
        self.lr_model.fit(self.train_fps, self.classes_train)

         
    
    def lr_model_predict(self):
        self.class_preds = self.lr_model.predict(self.test_fps)

         




    def check_se(self, fps):
        """ Checks the change in shannon entropy of a given molecule 'm' against
            a set of active molecules (actives_np_fps) """

        frequencies = []
        se_i = []

        total_len = float(len(fps))

        # Get the number of 1s
        ones_dict = self.get_index_totals(fps)
        ones = ones_dict.values()

        # Calculate frequency of 1s in every position
        for x in ones:
            frequencies.append(float(x/total_len))

        for x in frequencies:
            if x in {0,1}:
                se = 0
            else:
                # Use Shannon Entropy definition as detailed in Publication
                se = -x*math.log(x,2) - (1-x)*math.log(1-x,2)

            se_i.append(se)

        se_t = sum(se_i)

        return se_t


    def get_index_totals(self, fps):
        """ Counts the number of '1' bits at every FP index of a given mol """
        se_i = []

        indexes = [x for x in range(2048)]
        idx_counts = {key: 0 for key in indexes}

        for i in range(0,2048):
            for fp in fps:
                if fp[i] == 1:
                    idx_counts[i] += 1

        return idx_counts

    def split_data(self, data):
        fps = []
        temp = []

        potency = [math.log(float(x[0]),10) for x in data]
        assay_classification = [x[1] for x in data]
        molset = [x[2] for x in data]
        smiles = [x[3] for x in data]
        ids = [x[4] for x in data]

        mols = [Chem.MolFromSmiles(x) for x in smiles]

        fps = self.generate_fingerprints(smiles)

        return potency, assay_classification, smiles, ids, fps

    def generate_fingerprints(self, smiles):
        temp = []
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        bit_info = {}
        fp = [AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=2048, bitInfo=bit_info) for x in mols]

        for arr in fp:
            temp.append([int(x) for x in arr])
        fps = numpy.array(temp, dtype=float)

        return fps

    def get_model_weights(self):
        weight_dict = {}

        for layer in self.model.layers:
            weights = layer.get_weights()
            i = 1
            for x in weights[0]:
                weight_dict[i] = numpy.sum(x)
                i += 1

            self.sorted_weights = sorted(weight_dict.items(), key=operator.itemgetter(1))


    def load_model(self):
        self.model = load_model(self.model_filename)

    def create_model(self):

        #print "Creating sequential model"
        self.model = Sequential()

        self.model.add(Dense(2048, input_dim=2048, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dropout(0.05, input_shape=(2048,)))
        self.model.add(Dense(2048, input_dim=2048, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dropout(0.05, input_shape=(2048,)))
        self.model.add(Dense(2048, input_dim=2048, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal'))
        self.model.compile(loss='mean_absolute_error', optimizer='Adam')

        return self.model

    def fit_model(self):
        #print ""
        print("Fitting sequential model")

        #f = 20/len(self.train_potency)
        #batches = int(len(self.train_potency)/15)

        #fraction = self.train_fraction/10
        self.model.fit(self.train_fps, self.train_potency, epochs=200, verbose=0)

    def save_sequential_model(self):
        #print ""
        print("Saving sequential model")
        self.model.save('sequential_model.h5')

    def model_predict(self):
        self.predictions = self.model.predict(self.test_fps, verbose=0)

        return self.predictions

    def write_csv(self):
        with open('regression-accuracy.csv', 'a') as f:

            row = '{}\n'.format(self.average)
        #for item in self.average:
            f.write(row)


    def model_accuracy(self):
        self.final_predictions = self.predictions.flatten()
        temp = list(self.final_predictions)
        diff = []

        for i,x in enumerate(self.final_predictions):
            diff.append(abs(x-self.test_potency[i]))

        self.average = sum(diff) / float(len(diff))

        print("Average MUE for this run: {}".format(self.average))

        return self.average


def split_types(smiles, ic50):

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    print("total num of examples:{}".format(len(ic50)))
    print("")
    print("splitting data...")

    test_split=float(0.999)
    x_train, x_test, y_train, y_test = train_test_split(smiles,ic50,test_size=test_split)

    #print ""
    print("There are a total of {} training molecules ".format(len(x_train)))
    print("There are a total of {} testing molecules ".format(len(x_test)))

    #print "AVERAGE IC5O TRAIN:{}".format(reduce(lambda x, y: x + y, y_train) / len(y_train))
    templist = [float(x) for x in y_train]
    spread = max(templist) - min(templist)
    print("The spread is {}: ".format(spread))
    print("std: {}".format(numpy.std(templist)))


    return x_train, x_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser(description='options parser for ML regression of OSM data')
    #parser.add_argument('--model_file', dest="model_filename")
    parser.add_argument('--input_file', dest="input_filename")
    parser.add_argument('--test_file', dest='test_filename')
    args = parser.parse_args()


    classes = [] 

    # Retrieve the data
    train_filename = args.input_filename
    test_filename = args.test_filename 

    train_df = pandas.read_csv(train_filename)
    test_df = pandas.read_csv(test_filename)
    
    nm = [float(x*1000) for x in train_df['ic50'].values.tolist()]
    train_df['ic50_nm'] = nm
    diff = []

    for i,x in enumerate(train_df['ic50_nm'].values.tolist()):
        if x <= 1000:
            classes.append(1)
        else:
            classes.append(0)
    
    train_df['classes'] = classes 

    smiles_train = train_df['smiles'].values.tolist()
    ic50_train = train_df['ic50_nm'].values.tolist()
    classes_train = train_df['classes'].values.tolist()

    id_test = test_df['OSM Code']
    smiles_test = test_df['SMILES'].values.tolist()
    run = Regression(smiles_train, smiles_test, ic50_train, classes_train, id_test)

    #ic50_results = [10**x[0] for x in run.potency_predictions]
    #print(ic50_results)
    #class_results = run.class_preds
    #print(class_results)

if __name__ == '__main__':
    main()
