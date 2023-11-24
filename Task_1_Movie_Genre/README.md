# Movie Genre Classification

This repository contains code for movie genre classification using Multinomial Naive Bayes with TF-IDF vectorization. The goal is to predict the genre of a movie based on its title and plot.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## Usage

- convert txt files into csv files.
- Load and preprocess the training and testing data.
- Combine 'title' and 'plot' into a single text column.
- Vectorize the text using TF-IDF (Term Frequency-Inverse Document Frequency).
- Train the Multinomial Naive Bayes classifier.
- Evaluate the model on the validation set.
- Make predictions on the test set.

## Dataset
The training and testing datasets are assumed to be in CSV format with columns 'title', 'plot', and 'genre'. Ensure you have the correct file paths for the training and testing datasets.
link- [https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb]

## Requirements
Make sure to install the required Python packages.
