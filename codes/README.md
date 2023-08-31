# Baseline codes

We have designed three baseline models that can utilize either BERT or CNN as the encoder.

* **AutoEncoder** AutoEncoder (AE) is extensively employed in anomaly detection, in which the key idea is to train the AE model with normal sample data.
The model compresses the input data into a lower-dimensional latent space and then attempts to reconstruct the data back into the original space by a decoder. 
During the testing phase, the model calculates the reconstruction loss for the test samples.
* **VAE** VAE is a variant of AE that incorporates the concept of probabilistic models. 
In contrast to AE, VAE utilizes an encoder to generate the parameters of a posterior probability distribution and sample latent variables from this distribution. This allows for a more flexible capture of the sample features.
* **GANomaly** The core idea of GANomaly is to jointly learn the generation of high-dimensional space and the inference of latent space through a conditional generative adversarial network. 
During the training process, minimizing the distance between latent vectors in the generator network helps learn the feature distribution of normal samples. 
When inference, a larger distance metric from the learned data distribution indicates an anomaly in that distribution.

## Requirements and Installation
python3  
```
pip install torch==1.10.1
pip install transformers==4.29.2
pip install scikit-learn==1.2.2
```

## Preprocessing data
put the dataset into data folder and run the data_split.py.
```
cd data
python data_split.py
```

## Training & Testing
```
python main.py
```
