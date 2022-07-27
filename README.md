# Generate Faces

A Generative Adverserial Network (GAN) produces fake celebrity faces. From a dataset of over 200,000 low resolution pictures that are cropped to 28x28, this machine learning architecture learns to generate images and discriminate it's own creation.

## Getting Started

Simply run the Jupyter Notebook dlnd_face_generation.ipynb or you can run the script face_generation.py

```
python face_generation.py
```

## I Just Want to See the Results!
Just open `dlnd_face_generation.ipynb` here on Github and you can view the results of the project!

### Prerequisites

You can install the required packages through Anaconda's environment manager using the machine-learning.yml file

```
conda env create -f machine-learning.yml
```

Then, activate the environment and run face_generation.py

```
activate machine-learning
```

Otherwise, check out the machine-learning.yml file for dependencies and their versions

## Running the tests

Simply add test cases to problem_unittests.py or run it

```
python problem_unittests.py
```

## Built With

* [TensorFlow](https://www.tensorflow.org/install/install_windows) - The machine learning framework
* [Anaconda](https://repo.continuum.io/archive/Anaconda3-5.1.0-Windows-x86_64.exe) - The environment manager
* [Jupyter Notebook](http://jupyter.org/install) - The code documentation
