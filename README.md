# ME395-Machine-Learning-for-Mechanical-Sciences
This repository contains the associated files for the final project of MECH_ENG 395 at Northwestern University. It explores a recurrent sequence-to-sequence model for arterial blood pressure waveform reconstruction from PPG to ECG signals


---
## Folder Description
* deepLSTM_pytorch.ipynb: The main file where the machine learning models are built, trained, and evaluated. The code for results visualization is also included here.
* data_loader.py: File used to load the MIMIC-III dataset. Includes some preprocessing that synchronizes signals into 4-second windows. Modified from https://github.com/powhwee/bp-measure/tree/main
* trained models
  * deepLSTM_baseline.pt: saved trained weights and parameters for the baseline model
  * deepLSTM_physics.pt: saved trained weights and parameters for the physics-informed model
* results/ : folder containing images of the results and comparisons of the deepLSTM network

---
## Software Environment
The model was developed and trained in Python using the PyTorch deep learning framework. The implementation was executed in Google Colab using the NVIDIA Tesla T4 GPU. The complete trainig and evaluation process required approximately 7 minutes of runtime.

---
## Dependencies
To build and run the model, the following dependencies are needed:
* Python 3.x
* numpy
* matplotlib
* PyTorch
  
Additional standard-library modules used:
* pathlib
* typing

---
## Execution Steps
1. Clone this repository
2. Data file is too large to include in this repo. Download the data from [this link](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation). For this project, only *Part_1.mat* was used.
3. Open *deepLSTM_pytorch.ipynb* in Google Colab
4. Move *Part_1.mat* and *data_loader.py* into a folder in your Google Drive titled **ME395**. Alternatively, you can modify the **Setup** cell of the Jupyter notebook to change file paths.
5. Use the **Configurations** cell to modify hyperparameters if needed.
6. Run all to execute the training and evaluation of both the baseline and physics-informed model.
7. Generated figures and performance metrics should reproduce the results reported in the project
