# Student Identity Classifier

A classifier to predict student identity of a certain author.

## Introduction

Train a LinearClassifier model or a DNNClassifier model to predict whether a certain scholar/author is a student or not based on his/her five statistics:
  * pc: number of total publications;
  * cn: number of total citations;
  * hi: H-index;
  * gi: G-index;
  * year_range: ime range from the first to the last publication. 
    * *year_range = year of latest publication - year of earlist publication + 1*

## Usage

```
import authorship_classifier
```

### Load data

Load training data from /data/train.csv (as training_path)

```
training_examples, training_targets, validation_examples, validation_targets = authorship_classifier.preprocess_data(
           training_path)
```

### Train LinearClassifier model

```
linear_classifier = authorship_classifier.train_linear_classification_model(
        learning_rate = 0.1,
        regularization_strength = 0,
        batch_size= 20,
        steps = 1000,
        training_examples = training_examples,
        training_targets = training_targets,
        validation_examples = validation_examples,
        validation_targets = validation_targets)    
```

### Train DNNClassifier model

```
dnn_classifier = authorship_classifier.train_nn_classification_model(
        learning_rate = 0.1,
        regularization_strength = 0,
        batch_size= 20,
        hidden_units = [10, 10],
        steps = 1000,
        training_examples = training_examples,
        training_targets = training_targets,
        validation_examples = validation_examples,
        validation_targets = validation_targets,
        remove_event_files = False)
```

### Test model

Load test data from /data/test.csv (as test_path)

```
authorship_classifier.test_classifier(dnn_classifier, test_path)
```

*(Recommended) Set Tensorflow verbosity to only errors before training:*

```
tf.logging.set_verbosity(tf.logging.ERROR)
```

## Requirements

The following environment or packages need to be properly installed.

```
python >= 3.6
tensorflow >= 1.10.0
pandas >= 0.23.4
numpy >= 1.15.2
seaborn >= 0.9.0
scikit-learn >= 0.20.0
matplotlab >= 3.0.0
```

## Authors

* **Junlin Song** - *Initial work* - [julius-song](https://github.com/julius-song)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
