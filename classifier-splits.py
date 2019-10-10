import csv
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import math
import operator
import numpy
import argparse
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
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
class Classifier(object):
    def __init__(self, smiles_train, smiles_test, class_train, class_test):


        # Split data into necessary groups
        self.train_smiles = smiles_train
        self.train_fps = self.generate_fingerprints(smiles_train)
        self.class_train = class_train 

        self.test_smiles = smiles_test
        self.test_fps = self.generate_fingerprints(smiles_test)
        self.class_test = class_test 

        # Compile the Sequential Model1
        self.create_svm_model()

        # Train the model
        self.fit_model()

        # Make predictions
        self.model_predict()

        # Determine the MUE, and the internal metric for distance, and fp coverage
        self.model_accuracy()



    def generate_fingerprints(self, smiles):
        temp = []
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        bit_info = {}
        fp = [AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=2048, bitInfo=bit_info) for x in mols]

        for arr in fp:
            temp.append([int(x) for x in arr])
        fps = numpy.array(temp, dtype=float)

        return fps


    def load_model(self):
        self.model = load_model(self.model_filename)

    def create_svm_model(self):
        self.model = SVC(kernel='linear')

    def create_lr_model(self):
        self.model = LogisticRegression(solver='liblinear')

        return self.model

    def fit_model(self):
      
        self.model.fit(self.train_fps, self.class_train)


    def model_predict(self):
        self.predictions = self.model.predict(self.test_fps)
        return self.predictions


    def model_accuracy(self):
        self.final_predictions = self.predictions.flatten()
        temp = list(self.final_predictions)

        self.MCC = matthews_corrcoef(self.class_test, temp)

        #print(self.MCC)

def main():
    parser = argparse.ArgumentParser(description='options parser for ML regression of OSM data')
    parser.add_argument('--input_file', dest="input_filename")
    args = parser.parse_args()

    # Retrieve the data
    filename = args.input_filename
    df = pandas.read_csv(filename)
    nm = [float(x*1000) for x in df['ic50'].values.tolist()]
    df['ic50_nm'] = nm
    classes = []

    # Create class data based in ic50: 1 for active, 0 for inactive w/ 1000nM cutoff 
    for val in df['ic50_nm'].values.tolist():
        if float(val) < 1000:
            classes.append(1)
        else:
            classes.append(0)
    df['classes'] = classes 
    
    print('There are a total of {} active compounds'.format(classes.count(1)))
    print('There are a total of {} inactive compounds'.format(classes.count(0)))
    mcc = []
    # Check potency predictions 10 times, 15% test size (small sample sizs)
    for i in range(100):
        mask = numpy.random.rand(len(df)) < 0.7
        
        train = df[mask]
        test = df[~mask]

        smiles_train = train['smiles'].values.tolist()
        class_train = train['classes'].values.tolist()

        smiles_test = test['smiles'].values.tolist()
        class_test = test['classes'].values.tolist()

        train_one_count = class_train.count(0)
        test_one_count = class_test.count(0)
        if train_one_count > 0:
            if test_one_count > 0:
                run = Classifier(smiles_train, smiles_test, class_train, class_test)
                mcc.append(run.MCC)
    
    print('Mean MCC : {}'.format(numpy.mean(mcc)))
    print('STD MCC : {}'.format(numpy.std(mcc)))

if __name__ == '__main__':
    main()
