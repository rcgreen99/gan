# GAN

A simple implementation of a Generative Adversarial Network (GAN) using PyTorch.

# Requirements

This project uses [uv](https://docs.astral.sh/uv/), a modern package manager for Python. Click the link to for installation instructions.


# Usage

### Training

To train a model, use the command:

`bin/train.sh`

This will automatically download the MNIST dataset and train the model. Outputs will be saved in the `logs` directory.

### Generating Images

To generate images, use the command:

`bin/generate.sh --generator_path <path_to_generator_checkpoint>`

This will load the generator checkpoint and generate 16 images. The generated images will be saved in the `generated` directory.
