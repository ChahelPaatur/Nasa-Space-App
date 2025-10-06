Exoplanet Detection Paradigm
Overview

This project explores how machine learning models can be applied to exoplanet detection using NASA’s exoplanet archive. The system classifies and identifies exoplanets using descriptive features, supporting the broader mission of understanding planetary formation, searching for Earth-like planets, and enabling potential future exploration.

Data Source

We used the NASA Exoplanet Archive, which provides:

Planet name
Orbital period
Discovery year
Other measurable properties of the planet
Methodology

Our approach combined multiple AI architectures:

CNN (Convolutional Neural Network): Used for filtering data and classification tasks such as image and object detection.
RNN (Recurrent Neural Network): Applied to sequential data like text and time-series information.
DRL (Deep Reinforcement Learning): An agent-based model leveraging trial-and-error learning.
Hybrid Architecture: Integration of CNN, RNN, and DRL with backpropagation for improved accuracy.
Implementation
The AI models were developed in Python.
The front-end and user interaction were built primarily using HTML, with API fetching for efficient data access, improved error handling, and secure resource management.
AI was used minimally on the DNQ network itself, with the majority of AI implementation focused on the front-end process to enhance accessibility and usability.
Model Saving

The trained models were saved using standard TensorFlow/Keras workflows:

# Saving a trained model
model.save("exoplanet_model.h5")

# Loading the saved model
from tensorflow.keras.models import load_model
model = load_model("exoplanet_model.h5")


This ensures reproducibility and allows continued training or deployment without retraining from scratch.

Mission

Our goal is to leverage AI to help astronomers and researchers:

Identify potential Earth-like exoplanets
Support discoveries about planetary formation and evolution
Enable insights into possible future colonization
Acknowledgments
NASA Exoplanet Archive for the dataset
Project developed under Mountain House to the Sky, June 2025
