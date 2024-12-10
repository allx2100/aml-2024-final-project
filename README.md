# Variational Autoencoders and Normalizing Flows for Anomaly Detection
CS 6362 | Final Project  
Alexander Lin & Allan Zhang

## Set Up
1. Create a Python virtual environment.
2. Run ```pip install -r requirements.txt``` to load to required modules.
3. Run ```pip install -e .``` to allow imports across directories.

## Data Acquisition
### CIFAR-10
1. Run ```wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz``` in the ```data``` directory and untar the downloaded file.
2. Execute ```data/join_data.py```,```data/assemble_sets.py``` and then ```data/assemble_dataloaders.py``` to create the PyTorch DataLoaders necessary for model training and testing.

### MNIST
1. Download MNIST data from ```https://drive.google.com/file/d/11ZiNnV3YtpZ7d9afHZg0rtDRrmhha-1E/view``` into the ```data``` directory.
2. Execute the ```data/assemble_sets_mnist.py``` script to create the MNIST data used for anomaly testing.

## Model Training
Run any of the training Python scripts to train a given model. The hyperparameters of the model can be edited by parameterizing the model constructor.

## Results
### Experiment 1
```anomaly.py``` and ```latent_graph.py``` were used to generate Figure 1 in the paper. Substitute any of models labeled 1 in ```final_models``` into the script to review their specific results. These may vary from what is presented in the paper since it is dependent on randomized data.

### Experiment 2
```anomaly_cifar.py``` and ```latent_graph_cifar.py``` were used to generate Figure 1 in the paper. Substitute any of models labeled 2 in ```final_models``` into the script to review their specific results. These may vary from what is presented in the paper since it is dependent on randomized data.