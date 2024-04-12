#!/bin/bash

# Step 1: Generate data
python data_creation.py

# Step 2: Preprocess data
python model_preprocessing.py

# Step 3: Train the model
python model_preparation.py

# Step 4: Test the model
python model_testing.py