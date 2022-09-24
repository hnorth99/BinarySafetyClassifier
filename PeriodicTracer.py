import pandas as pd
import numpy as np
import os
import statistics
import csv
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random


class PeriodicTracer:
    def __init__(self, training_data_path, testing_data_path, exp_name):
        if not os.path.exists(exp_name):
            os.mkdir(exp_name)
        if not os.path.exists(exp_name + '/checkpoints'):
            os.mkdir(exp_name + '/checkpoints')
        if not os.path.exists(exp_name + '/graphics'):
            os.mkdir(exp_name + '/graphics')
        self.training_data_path = training_data_path
        self.testing_data_path = testing_data_path
        self.exp_name = exp_name
        self.root = exp_name + '/'

    def __upsample_classes(self, raw_df):
        classes = raw_df['target'].unique()

        max_class_count = 0
        for c in classes:
            if len(raw_df[raw_df['target'] == c]) > max_class_count:
                max_class_count = len(raw_df[raw_df['target'] == c])

        balanced_df = pd.DataFrame()
        for c in classes:
            if len(balanced_df) > 0:
                others = balanced_df[balanced_df['target'] != c]
            else:
                others = pd.DataFrame()

            upsample = resample(raw_df[raw_df['target'] == c], replace=True, n_samples=max_class_count)
            balanced_df = pd.concat([others, upsample])
        return balanced_df

    # Turn the labels in a dataset into a dataframe with mean and var of every word
    def __mean_var_df(self, data_path, n, m, balance_classes=False, testing=False):
        # Raw data
        raw_df = pd.read_csv(data_path)
        # Balanced data
        if not testing and balance_classes:
            raw_df = self.__upsample_classes(raw_df)

        # Generate all the possible terms that exist at n, m dimension
        terms = self.__generate_term_list(n, m)

        # Create the columns that will be in our new csv file
        cols = ['label', 'target']
        for term in terms:
            cols.append(term + '_mean')
            cols.append(term + '_var')

        # Create a csv file to dump data into
        filepath = self.exp_name + '/checkpoints/mean_var_df/m' + str(m) + 'n' + str(n)
        if testing:
            filepath += 'test'
        if not os.path.exists(self.exp_name + '/checkpoints/mean_var_df'):
            os.mkdir(self.exp_name + '/checkpoints/mean_var_df')
        with open(filepath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(cols)

        # Analyze one binary file at a time
        for entry in raw_df.itertuples(index=True, name='Pandas'):
            # Row that will hold mean_var data specific to current binary file
            row = []
            # Variables for entry specifics
            label = entry[1]
            target = entry[2]
            row.append(label)
            row.append(target)

            # Use a dictionary that will map a term to all the indices of its occurence
            term_to_indices = {}
            for term in terms:
                term_to_indices[term] = []

            # Move word by word through raw_df file updating the term to index dictionary
            for i in range(len(label) - n + 1):
                # Current selection in binary file
                phrase = label[i:i + n]
                # Figure out what terms are in the selected phrase
                found_terms = self.__get_terms_in_phrase(phrase, m)
                # Update dictionary
                for term in found_terms:
                    term_to_indices[term].append(i)

            # Use a dictionary that will map a term to the number of spacing occuring between every two occurences
            # of the term
            term_to_spacings = {}
            for term in terms:
                term_to_spacings[term] = []

            # Cycle through dictionary converting term indices into term spacings
            for term in terms:
                # Reduce indices to spacings
                indices = term_to_indices[term]
                for i in range(len(indices) - 1):
                    spacing = indices[i + 1] - indices[i]
                    term_to_spacings[term].append(spacing)
                # Determine mean of spacings
                term_mean = 0
                if len(term_to_spacings[term]) > 0:
                    term_mean = statistics.mean(term_to_spacings[term])
                    # Determine variance of spacings
                term_variance = 0
                if len(term_to_spacings[term]) > 1:
                    term_variance = statistics.variance(term_to_spacings[term])
                # Append the mean and variance to the row
                row.append(term_mean)
                row.append(term_variance)

            # Open file in append mode
            with open(filepath, 'a', newline='') as f:
                # Create a writer object from csv module
                writer = csv.writer(f)
                # Add contents of list as last row in the csv file
                writer.writerow(row)
                f.close()

        # Read the csv file into a pandas dataframe
        pt_v1_df = pd.read_csv(filepath)
        # Return the dataframe
        return pt_v1_df

    # Turn the labels in a dataset into a dataframe with mean and var of every word (training data)
    def __training_mean_var_df(self, n, m, balance_classes=False):
        return self.__mean_var_df(self.training_data_path, n, m, balance_classes=balance_classes)

    # Turn the labels in a dataset into a dataframe with mean and var of every word (testing data)
    def __testing_mean_var_df(self, n, m):
        return self.__mean_var_df(self.testing_data_path, n, m, testing=True)

    # Return an array that represents all the possible strings/terms that can be created with n bits --
    # with m of these bits being "skips"
    def __generate_term_list(self, n, m):
        terms = []
        self.__generate_term_list_h(n - 1, m, "0", terms)
        self.__generate_term_list_h(n - 1, m, "1", terms)
        return terms

    # Recursive helper method for generate_term_list
    def __generate_term_list_h(self, n, m, term, terms):
        # Used all space and all skips
        if n == 0 and m == 0:
            terms.append(term)
        # At last space or have more space but no more skips
        elif n == 1 or (n > 0 and m == 0):
            self.__generate_term_list_h(n - 1, m, term + "0", terms)
            self.__generate_term_list_h(n - 1, m, term + "1", terms)
        # Have enough space to include the required skips
        elif n - m >= 1:
            self.__generate_term_list_h(n - 1, m, term + "0", terms)
            self.__generate_term_list_h(n - 1, m, term + "1", terms)
            self.__generate_term_list_h(n - 1, m - 1, term + "_", terms)

    # Return the list of all sub-terms existing within a given phrase if you insert m skips
    def __get_terms_in_phrase(self, phrase, m):
        found_terms = []
        self.__get_terms_in_phrase_h(phrase, found_terms, phrase[0], m, 1)
        return found_terms

    # Recursive helper method for get_terms_in_phrase
    def __get_terms_in_phrase_h(self, phrase, found_terms, term, m, i):
        if i == len(phrase) - 1 and m == 0:
            found_terms.append(term + phrase[i])
        elif i < len(phrase) - 1:
            self.__get_terms_in_phrase_h(phrase, found_terms, term + phrase[i], m, i + 1)
            if m > 0:
                self.__get_terms_in_phrase_h(phrase, found_terms, term + "_", m - 1, i + 1)

    # Convert a mean var df into a pt3 df with omega weight w
    def __pt1_to_pt3(self, df, term, w):
        return df[term + '_mean'] / (1 + df[term + '_var'] ** w)

    # Convert a mean var df into a pt3 df using the added omega column in the df
    def __apply_omega_to_mean_var(self, df, term):
        return df[term + '_mean'] / (1 + df[term + '_var'] ** df[term + '_w'])

    # Get the optimal omega term for an n,m term of a given ISA
    def __get_optimal_omega(self, n, m, term):
        # Load a copy of mean_var df for modifcation
        filename = self.root + 'checkpoints/mean_var_df/m' + str(m) + 'n' + str(n)
        df = pd.read_csv(filename)

        # Generate how the spread of the ISA's term PT changes with different omega turns
        # Will make 50 samples of omega from 0 to 10
        samples, step = 50, 0.1
        sample_scores = np.zeros(samples)
        for i in range(samples):
            w = i * step
            df[term] = self.__pt1_to_pt3(df, term, w)

            # Generate the class cluster for each given class
            cs = df['target'].unique()
            class_cluster = 0
            c_means = np.zeros(len(cs))
            for j in range(len(cs)):
                # Get the mean of each class
                c = cs[j]
                c_pt3 = df.loc[df.target == c][term]
                c_mean = np.mean(c_pt3)
                c_means[j] = c_mean
                # Calculate the cluster for each class
                cluster = 0
                for pt in c_pt3:
                    cluster += abs(pt - c_mean)
                class_cluster += cluster

            # Generate the set cluster
            set_cluster = 0
            for j in range(len(c_means)):
                for k in range(j + 1, len(c_means)):
                    set_cluster += abs(c_means[j] - c_means[k])
            sample_scores[i] = set_cluster / class_cluster

        # Save fig
        fig = plt.figure()
        x = range(samples)
        plt.scatter(x, sample_scores)
        plt.savefig(self.root + '/graphics/omega_search/' + term)
        plt.close()

        # Take the omega value that has the smallest spread
        optimal_w = (1 + np.argmax(sample_scores[1:])) * step
        return optimal_w

    # Generate the optimal omega values into a new column for every word in the set
    def __training_omega_search(self, n, m, use_omega=True):
        # Generate all terms at n, m dimension
        terms = self.__generate_term_list(n, m)
        # Will store all finalized omega weights in term_w_map
        # Maps term to omega weight
        term_w_map = {}

        # Load mean_var df
        filepath = self.root + 'checkpoints/mean_var_df/m' + str(m) + 'n' + str(n)
        pt = pd.read_csv(filepath)

        # Create directory for omega adjustment chart
        if not os.path.exists(self.root + '/graphics/omega_search'):
            os.mkdir(self.root + '/graphics/omega_search')
        plt.ioff()

        # Search for omega weights term by term
        for term in terms:

            if use_omega:
                term_w_map[term] = self.__get_optimal_omega(n, m, term)
            else:
                term_w_map[term] = 1

            # Add the omega terms into a new column of the dataframe
            w = np.zeros(len(pt))
            # Process binary file by file
            for i in range(len(pt)):
                # Add omega weight into the future w column
                w[i] = term_w_map[term]
            # Insert omega column into df
            pt[term + '_w'] = w

        # Create a csv file to dump data into
        location = self.root + 'checkpoints/'
        if not os.path.exists(location + 'omega_added/'):
            os.mkdir(location + 'omega_added/')
        if not os.path.exists(location + 'omega_weights/'):
            os.mkdir(location + 'omega_weights/')

        # Write omega applied df into a csv file
        filepath = location + 'omega_added/m' + str(m) + 'n' + str(n)
        pt.to_csv(filepath, index=False)

        # Save omega weights in json file
        filepath = location + 'omega_weights/m' + str(m) + 'n' + str(n) + '.json'
        with open(filepath, "w") as outfile:
            json.dump(term_w_map, outfile)

    # Add saved omega weights to testing data
    def __testing_omega_tack(self, n, m):
        # Load mean_var df
        filepath = self.root + 'checkpoints/mean_var_df/m' + str(m) + 'n' + str(n) + 'test'
        pt = pd.read_csv(filepath)

        # Load omega weights
        filepath = self.root + 'checkpoints/' + 'omega_weights/m' + str(m) + 'n' + str(n) + '.json'
        omega_weights = {}
        with open(filepath) as json_file:
            omega_weights = json.load(json_file)

        # Tack omega weights column term by term
        terms = self.__generate_term_list(n=n, m=m)
        for term in terms:
            # Add the omega terms into a new column of the dataframe
            w = np.zeros(len(pt))
            # Process binary file by file
            for i in range(len(pt)):
                # Add omega weight into the future w column
                w[i] = omega_weights[term]
                # Insert omega column into df
                pt[term + '_w'] = w
            # Insert omega column into df
            pt[term + '_w'] = w

        # Write omega applied df into a csv file
        filepath = self.root + 'checkpoints/omega_added/m' + str(m) + 'n' + str(n) + 'test'
        pt.to_csv(filepath, index=False)

    # Convert mean, var, and omega terms into final periodic tracing df
    def __apply_omega(self, n, m, testing=False):
        # Generate all terms at n, m dimension
        terms = self.__generate_term_list(n, m)
        # Read data in
        filepath = self.root + 'checkpoints/omega_added/m' + str(m) + 'n' + str(n)
        if testing:
            filepath += 'test'
        df = pd.read_csv(filepath)

        # Convert mean var and omega column term by term into pt3
        for term in terms:
            # Build the pt3 column for said term
            df[term] = self.__apply_omega_to_mean_var(df, term)
            # Drop no longer needed columns
            df.drop(term + '_var', axis=1, inplace=True)
            df.drop(term + '_mean', axis=1, inplace=True)
            df.drop(term + '_w', axis=1, inplace=True)

        # Create a csv file to dump data into
        location = self.root + 'checkpoints/omega_applied'
        if not os.path.exists(location):
            os.mkdir(location)
        filepath = location + '/m' + str(m) + 'n' + str(n)
        if testing:
            filepath += 'test'
        df.to_csv(filepath, index=False)
        return df

    # Convert mean, var, and omega terms into final periodic tracing df (training data)
    def __training_apply_omega(self, n, m):
        return self.__apply_omega(n, m)

    # Convert mean, var, and omega terms into final periodic tracing df (testing data)
    def __testing_apply_omega(self, n, m):
        return self.__apply_omega(n, m, testing=True)

    # Use training data to generate ensemble classifiers and save models into a file
    def __train_ensembles(self, n, m):
        df = pd.read_csv(self.root + 'checkpoints/omega_applied/m' + str(m) + 'n' + str(n))

        # Build set of features that will be used for predictions
        features = []
        for col in df.columns:
            if col != 'label' and col != 'target':
                features.append(col)

        # Break data into features and target
        X_train = df[features]
        y_train = df['target']

        # Train models
        # Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100)
        rf_clf.fit(X_train, y_train)
        # Gradient Boosting
        gb_clf = GradientBoostingClassifier(n_estimators=100)
        gb_clf.fit(X_train, y_train)
        # Extra Trees
        et_clf = ExtraTreesClassifier(n_estimators=100)
        et_clf.fit(X_train, y_train)

        # Save models
        # Create a csv file to dump data into
        location = self.root + 'models/'
        if not os.path.exists(location):
            os.mkdir(location)
        root = location + 'm' + str(m) + 'n' + str(n)
        pickle.dump(rf_clf, open(root + '_rf', 'wb'))
        pickle.dump(gb_clf, open(root + '_gb', 'wb'))
        pickle.dump(et_clf, open(root + '_et', 'wb'))

        return [rf_clf, gb_clf, et_clf]

    # Score the generated ensembles
    def __score_ensembles(self, n, m):
        # Load in all models corresponding to m,n dimension
        filename = self.root + 'models/m' + str(m) + 'n' + str(n)
        rf_clf = pickle.load(open(filename + '_rf', 'rb'))
        gb_clf = pickle.load(open(filename + '_gb', 'rb'))
        et_clf = pickle.load(open(filename + '_et', 'rb'))

        # Load testing data
        filename = self.root + 'checkpoints/omega_applied/m' + str(m) + 'n' + str(n) + 'test'
        df = pd.read_csv(filename)

        # Build set of features that will be used for predicitions
        features = []
        for col in df.columns:
            if col != 'label' and col != 'target':
                features.append(col)

        # Break data into features and target
        X_test = df[features]
        y_test = df['target']

        # Generate confusion matrices
        if not os.path.exists(self.root + '/graphics/confusion_matrix'):
            os.mkdir(self.root + '/graphics/confusion_matrix')

        rf_predictions = rf_clf.predict(X_test)
        rf_cm = confusion_matrix(y_test, rf_predictions, labels=rf_clf.classes_)
        rf_cm = rf_cm.astype('float') / rf_cm.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=rf_clf.classes_)
        disp.plot()
        plt.savefig(self.root + 'graphics/confusion_matrix/rf_m' + str(m) + 'n' + str(n))
        plt.close()

        et_predictions = et_clf.predict(X_test)
        et_cm = confusion_matrix(y_test, et_predictions, labels=et_clf.classes_)
        et_cm = et_cm.astype('float') / et_cm.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=et_cm, display_labels=et_clf.classes_)
        disp.plot()
        plt.savefig(self.root + 'graphics/confusion_matrix/et_m' + str(m) + 'n' + str(n))
        plt.close()

        gb_predictions = gb_clf.predict(X_test)
        gb_cm = confusion_matrix(y_test, gb_predictions, labels=gb_clf.classes_)
        gb_cm = gb_cm.astype('float') / gb_cm.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=gb_cm, display_labels=gb_clf.classes_)
        disp.plot()
        plt.savefig(self.root + 'graphics/confusion_matrix/gb_m' + str(m) + 'n' + str(n))
        plt.close()

        # Score models
        rf_acc = rf_clf.score(X_test, y_test)
        gb_acc = gb_clf.score(X_test, y_test)
        et_acc = et_clf.score(X_test, y_test)
        print('rf ' + str(rf_acc))
        print('gb ' + str(gb_acc))
        print('et ' + str(et_acc))

    # Go through entire model training process and then score models
    def build_test_model(self, n, m, balance_classes=False, use_omega=True, checkpoint=-1):
        if checkpoint <= 0:
            self.__training_mean_var_df(n=n, m=m, balance_classes=balance_classes)
        if checkpoint <= 1:
            self.__testing_mean_var_df(n=n, m=m)
        if checkpoint <= 2:
            self.__training_omega_search(n=n, m=m, use_omega=use_omega)
        if checkpoint <= 3:
            self.__testing_omega_tack(n=n, m=m)
        if checkpoint <= 4:
            self.__training_apply_omega(n=n, m=m)
        if checkpoint <= 5:
            self.__testing_apply_omega(n=n, m=m)
        if checkpoint <= 6:
            self.__train_ensembles(n=n, m=m)
        self.__score_ensembles(n=n, m=m)

    def __quick_score(self, df, term):
        # Split data frame
        X = df[term].to_numpy().reshape(-1, 1)
        y = df['target'].to_numpy().reshape(-1, 1).ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # train model
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)

        # test model
        y_pred = clf.predict(X_test)
        return metrics.accuracy_score(y_test, y_pred)

    # Build an important dataset from a range up to n length words with max m spaces (for training labels)
    def training_build_important_dataset(self, n_max, m_max, balance_classes=False, use_omega=True):
        # Ensure that all the correct/necessary data is created
        for m in range(0, m_max + 1):
            print("collecting data m: " + str(m))
            for n in range(m + 2, n_max + 1):
                print("collecting data n: " + str(n))
                if not os.path.exists(self.root + 'checkpoints/omega_applied/m' + str(m) + 'n' + str(n)):
                    self.build_test_model(n, m, balance_classes=balance_classes, use_omega=use_omega)

        # Build final dataset to add important data to
        df_temp = pd.read_csv(self.root + 'checkpoints/omega_applied/m0n2')
        df_final = df_temp[['label', 'target']].copy()

        # Search for important data
        unimportant = []
        for m in range(0, m_max + 1):
            print("searching data m: " + str(m))
            for n in range(m + 2, n_max + 1):
                print("searching data n: " + str(n))

                df = pd.read_csv(self.root + 'checkpoints/omega_applied/m' + str(m) + 'n' + str(n))
                terms = self.__generate_term_list(n, m)
                for term in terms:
                    # Create randomized orderings of term in questions
                    df[term + '_r1'] = random.sample(df[term].to_list(), len(df[term].to_list()))
                    df[term + '_r2'] = random.sample(df[term].to_list(), len(df[term].to_list()))
                    df[term + '_r3'] = random.sample(df[term].to_list(), len(df[term].to_list()))
                    # Score all the different columns
                    scores = np.zeros(4)
                    scores[0] = self.__quick_score(df, term)
                    for i in range(1, 4):
                        scores[i] = self.__quick_score(df, term + "_r" + str(i))
                        df.drop(term + '_r' + str(i), axis=1)
                    print(scores[0])
                    # See if true column passes parameters to be considered important
                    if np.argmax(scores) == 0 and scores[0] > 0.8:
                        df_final[term] = df[term]
                    else:
                        unimportant.append(term)
        # Write important dataframe into a csv
        output_path = self.root + 'checkpoints/boruta_important'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        df_final.to_csv(output_path + '/m' + str(m_max) + 'n' + str(n_max))

        # Print out unimportant terms
        print('Boruta searching finished!')
        print('These terms are unimportant:')
        print(unimportant)

    # Build an important dataset from a range up to n length words with max m spaces (for testing labels)
    def testing_build_important_dataset(self, n_max, m_max):
        # Build final dataset to add important data to
        df_temp = pd.read_csv(self.root + 'checkpoints/omega_applied/m' + str(m_max) + 'n' + str(n_max) + 'test')
        df_final = df_temp[['label', 'target']].copy()

        # Add important columns
        for m in range(m_max + 1):
            for n in range(m + 2, n_max + 1):
                df = pd.read_csv(self.root + 'checkpoints/omega_applied/m' + str(m) + 'n' + str(n) + 'test')
                for col in df.columns:
                    df_final[col] = df[col]

        # Write important dataframe into a csv
        output_path = self.root + 'checkpoints/boruta_important'
        df_final.to_csv(output_path + '/m' + str(m_max) + 'n' + str(n_max) + 'test')

    def score_boruta_important(self, n_max, m_max):
        # Load training data
        filename = self.root + 'checkpoints/boruta_important/m' + str(m_max) + 'n' + str(n_max)
        train_df = pd.read_csv(filename).iloc[:, 1:]
        # Load testing data
        filename = self.root + 'checkpoints/boruta_important/m' + str(m_max) + 'n' + str(n_max) + 'test'
        test_df = pd.read_csv(filename).iloc[:, 1:]

        # Build set of features that will be used for predicitions
        features = []
        for col in train_df.columns:
            if col != 'label' and col != 'target':
                features.append(col)

        # Split features and targets
        X_train = train_df[features]
        y_train = train_df['target']
        X_test = test_df[features]
        y_test = test_df['target']

        # Train models
        # Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100)
        rf_clf.fit(X_train, y_train)
        # Gradient Boosting
        gb_clf = GradientBoostingClassifier(n_estimators=100)
        gb_clf.fit(X_train, y_train)
        # Extra Trees
        et_clf = ExtraTreesClassifier(n_estimators=100)
        et_clf.fit(X_train, y_train)

        # Predictions
        rf_predictions = rf_clf.predict(X_test)
        print(metrics.accuracy_score(y_test, rf_predictions))
        gb_predictions = gb_clf.predict(X_test)
        print(metrics.accuracy_score(y_test, gb_predictions))
        et_predictions = et_clf.predict(X_test)
        print(metrics.accuracy_score(y_test, et_predictions))
