# CIL 2021, Project 3 - Road Segmentation

This repository was created for the course "Computational Intelligence Lab" at ETH ZÃrich. 
It contains code used to predict where roads lie in given aerial images.  

The code was written by the group 'Cillers' consisting of Mike Boss, Jonas Passweg and Jonathan Ehrat.


### Setup  
To run the code a running version of Python 3 is needed. The version we used is Python 3.7.4. Information about how to get Python on your system should be readily available on the internet.  
Additionally you will need some Python packages. These can be installed via Pip, the package installer for Python.  
The needed packages are found in the 'requirements.txt' file and can easily be installed with the command ```pip install -r requirements.txt```  


### Running the Code
To run the code there are various ways.

#### Reproducing the results in the report
Reproducing the model trained for the report simply run ```python cillers_best.py```

#### Interactive
To interactively play with the provided models there is an interactive jupyter notebook.
Start a notebook server with the lightning.ipynb notebook using ```jupyter-notebook cillers_best_notebook.ipynb```.  
This opens the notebook in your browser.  

#### Train and Predict with one model
If you wish to train and predict a specific model you can start a python script with ```python training_scripts/model.py```, where model.py is the filename of the model you want to train.

#### Testing all models on the Leonhard Cluster
If you would like to run and test all models on the Leonhard cluster, connect to the cluster and use ```python run.py``` to have each model submited as a job.


