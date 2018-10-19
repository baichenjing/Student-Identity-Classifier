#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:07:46 2018

@author: julius
"""

import os
import glob
import math

import tensorflow as tf
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.python.data import Dataset
import numpy as np


def preprocess_data(train_path, training_set_ratio = 0.9):
    '''Normalize input data and partition training & validation sets.
    
    Args:
        train_path: A 'str' of input training file path.
        training_set_ratio: A 'float', the ratio indicating how much of the input
            data should be training sets.
            
    Return:
        training_examples: A Pandas 'DataFrame' containing training features.
        training_targets: A Pandas 'DataFrame' containing training labels.
        validation_examples: A Pandas 'DataFrame' containing validation features.
        validation_targets: A Pandas 'DataFrame' containing validation labels.
    '''
    
    author_df = pd.read_csv(train_path, sep = ',')
    
    author_normalized_df = normalize(author_df)

    # Reindex to eliminate influence of data orders.
    author_normalized_df = author_normalized_df.reindex(
            np.random.permutation(author_normalized_df.index))
    
    length = len(author_normalized_df.index)
    training_num = int(training_set_ratio * length)
    validation_num = length - training_num
    
    training_examples = process_features(author_normalized_df.head(training_num))
    training_targets = process_targets(author_normalized_df.head(training_num))
    
    validation_examples = process_features(author_normalized_df.tail(validation_num))
    validation_targets = process_targets(author_normalized_df.tail(validation_num))   
    
    return training_examples, training_targets, validation_examples, validation_targets
    

def normalize(examples_df):
    '''Normalize input data.
    
    Apply log normalization to feature 'pc', 'cn', 'hi', 'gi', 
    and linear normalization to feautre 'year_range'.
    
    Args:
        examples_df: A Pandas 'DataFrame' containing original features and labels.
        
    Return:
        A Pandas 'DataFrame' containing normalized features and original labels.
    '''
    
    normalized_df = pd.DataFrame()

    for feature in ['pc', 'cn', 'hi', 'gi']:
        normalized_df[feature] = examples_df[feature].apply(
                lambda val: math.log(val + 1.0))

    normalized_df['year_range'] = examples_df['year_range'].apply(
            lambda val: val / examples_df['year_range'].max())

    normalized_df['label'] = examples_df['label']
  
    return normalized_df


def process_features(df):
    '''Prepare input features .
    
    Args:
        df: A Pandas 'DataFrame' containing input features and labels.
        
    Return:
        A 'DataFrame' containing input features only.
    '''
    
    return df[['pc', 'cn', 'hi', 'gi', 'year_range']]


def process_targets(df):
    '''Prepare input features .
    
    Args:
        df: A Pandas 'DataFrame' containing input features and labels.
        
    Return:
        A 'DataFrame' containing input labels only.
    '''
    
    return df[['label']]


def construct_feature_columns(examples_dataframe):
    """Construct the TensorFlow Feature Columns.
    
    Returns:
        A set of feature columns
    """ 
    
    return set([tf.feature_column.numeric_column(my_feature) 
                for my_feature in examples_dataframe])
 
    
def my_input_fn(features, labels, batch_size = 1, num_epochs = None, shuffle = True):
    """Trains a linear or nn regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
      
    Returns:
      Tuple of (features, labels) for next data batch
    """

    features = {key: np.array(value) for key, value in dict(features).items()}
    
    ds = Dataset.from_tensor_slices((features, labels))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds.shuffle(10000)
        
    features, labels = ds.make_one_shot_iterator().get_next()
    
    return features, labels


def train_linear_classification_model(
        learning_rate,
        steps,
        batch_size,
        regularization_strength,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,
        remove_event_files = True,
        model_dir = os.path.join(os.getcwd(), 'linear_model')
    ):
    """Trains a linear classification model.
  
    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.
  
    Args:
        learning_rate: An `float`, the learning rate to use.
        steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        regularization_strength: A 'float', strength of l1 regularization.
        training_examples: A `DataFrame` containing the training features.
        training_targets: A `DataFrame` containing the training labels.
        validation_examples: A `DataFrame` containing the validation features.
        validation_targets: A `DataFrame` containing the validation labels.
        remove_event_files: True or False. Whether to remove tf events files.
      
    Returns:
        A `LinearClassifier` object trained on the training data.
    """
    
    periods = 10
    steps_per_period = int(steps / periods)
    
    # Create model directory.
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # Create a LinearClassifier object.
    my_optimizer = tf.train.FtrlOptimizer(
            learning_rate = learning_rate,
            l1_regularization_strength = regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
            my_optimizer, 5.0)
    
    classifier = tf.estimator.LinearClassifier(
            optimizer = my_optimizer,
            feature_columns = construct_feature_columns(training_examples),
            model_dir = model_dir
            )
    
    # Create input functions.
    training_input_fn = lambda: my_input_fn(
            training_examples, training_targets, batch_size = batch_size)
    predict_training_input_fn = lambda: my_input_fn(
            training_examples, training_targets, num_epochs = 1, shuffle = False)
    predict_validation_input_fn = lambda: my_input_fn(
            validation_examples, validation_targets, num_epochs = 1, shuffle = False)
    
    # Train the model.
    print('Training LinearClassifier model...')
    print('LogLoss error (on valdiaiton data):')
    
    training_errors = []
    validation_errors = []
    
    for period in range(periods):
        
        classifier.train(
                input_fn = training_input_fn,
                steps = steps_per_period)
        
        # Compute loss.
        training_predictions = classifier.predict(input_fn = predict_training_input_fn)
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 2)
        
        validation_predictions = classifier.predict(input_fn = predict_validation_input_fn)
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 2)
        
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        
        # Print current loss.
        print('\tperiod %02d: %0.2f' % (period, validation_log_loss))
        
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
        
    print('Model training finished.')
    
    if remove_event_files == True:
        # Remove event files to save disk space.
        _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))
  
    # Evaluate classifier.
    evaluation_metrics = classifier.evaluate(input_fn = predict_validation_input_fn)
    
    print('Final metrics (on validation data):')
    for metric in ['accuracy', 'auc', 'precision', 'recall']:
        print('{}: {:.3f}'.format(metric, evaluation_metrics[metric]))

    # Output a graph of loss metrics over periods.
    plt.figure(figsize = (5, 5))
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()
      
    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, validation_pred_class_id)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = (5, 5))
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
    return classifier


def train_nn_classification_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        regularization_strength,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,
        remove_event_files = True,
        model_dir = os.path.join(os.getcwd(), 'dnn_model')
        ):
    """Trains a neural network classification model for the authorship judgement.
  
    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.
  
    Args:
        learning_rate: An `float`, the learning rate to use.
        steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        hidden_units: A `list` of int values, specifying the number of neurons in each layer.
        regularization_strength: A 'float', strength of l1 regularization.
        training_examples: A `DataFrame` containing the training features.
        training_targets: A `DataFrame` containing the training labels.
        validation_examples: A `DataFrame` containing the validation features.
        validation_targets: A `DataFrame` containing the validation labels.
        remove_event_files: True or False. Whether to remove tf events files.
        
    Returns:
        The trained `DNNClassifier` object.
    """
    
    periods = 10
    steps_per_period = int(steps / periods)
    
    # Create model directory.
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # Create a DNNClassifier object.
    my_optimizer = tf.train.FtrlOptimizer(
            learning_rate = learning_rate,
            l1_regularization_strength = regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
            my_optimizer, 5.0)
    
    classifier = tf.estimator.DNNClassifier(
            optimizer = my_optimizer,
            hidden_units = hidden_units,
            feature_columns = construct_feature_columns(training_examples),
            config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1),
            model_dir = model_dir)
    
    # Create input functions.
    training_input_fn = lambda: my_input_fn(
            training_examples, training_targets, batch_size = batch_size)
    predict_training_input_fn = lambda: my_input_fn(
            training_examples, training_targets, num_epochs = 1, shuffle = False)
    predict_validation_input_fn = lambda: my_input_fn(
            validation_examples, validation_targets, num_epochs = 1, shuffle = False)
    
    # Train the model.
    print('Training DNNClassifier model...')
    print('LogLoss error (on valdiaiton data):')
    
    training_errors = []
    validation_errors = []
    
    for period in range(periods):
        
        classifier.train(
                input_fn = training_input_fn,
                steps = steps_per_period)
        
        # Compute loss.
        training_predictions = classifier.predict(input_fn = predict_training_input_fn)
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 2)
        
        validation_predictions = classifier.predict(input_fn = predict_validation_input_fn)
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 2)
        
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        
        # Print current loss.
        print('\tperiod %02d: %0.2f' % (period, validation_log_loss))
        
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
        
    print('Model training finished.')
    
    if remove_event_files == True:
        # Remove event files to save disk space.
        _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))
    
    # Evaluate classifier.
    evaluation_metrics = classifier.evaluate(input_fn = predict_validation_input_fn)
    
    print('Final metrics (on validation data):')
    for metric in ['accuracy', 'auc', 'precision', 'recall']:
        print('{}: {:.3f}'.format(metric, evaluation_metrics[metric]))

    # Output a graph of loss metrics over periods.
    plt.figure(figsize = (5, 5))
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()
      
    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, validation_pred_class_id)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = (5, 5))
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
    return classifier


def test_classifier(classifier, test_path):
    '''Test performance of the trained classifier.
    
    Args:
        classifier: A trained 'LinearClassifier' or 'DNNClassifier' object.
        test_path: A 'str' of input test file path.
    '''
    
    test_df = pd.read_csv(test_path, sep = ',')
    
    test_normalized_df = normalize(test_df)

    test_examples = process_features(test_normalized_df)
    test_targets = process_targets(test_normalized_df)
    
    test_input_fn = lambda: my_input_fn(test_examples, test_targets, num_epochs = 1, shuffle = False)
    
    # Evaluate classifier on test data.
    evaluation_metrics = classifier.evaluate(input_fn = test_input_fn)
    
    print('Final metrics (on test data):')
    for metric in ['accuracy', 'auc', 'precision', 'recall']:
        print('{}: {:.3f}'.format(metric, evaluation_metrics[metric]))

    # Output a plot of the confusion matrix.
    test_predictions = classifier.predict(input_fn = test_input_fn)
    test_pred_class_id = np.array([item['class_ids'][0] for item in test_predictions])
    
    cm = metrics.confusion_matrix(test_targets, test_pred_class_id)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = (5, 5))
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()    
    
