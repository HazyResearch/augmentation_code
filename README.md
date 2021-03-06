# Augmentation

## Dependencies
- Python 3.6+
- Pillow, matplotlib, numpy, pytorch==0.4.0, seaborn, torchvision

## Usage

* `mnist_experiments.py` runs a full set of experiments on MNIST and save the
    results to the directory `saved`. Note: the default run take a long time (43 hours) to
    finish, since we're running for all 10 seeds.
    ```
    python mnist_experiments.py
    ```

    Currently, it executes the following experiments:
    1. Measure the difference between exact augmented objective and approximate
       objectives (on original images, 1st order approximation, 2nd order approximation).
    2. Measure the agreement and KL divergence between the predictions made by
       model trained on exact augmented objective and models trained on
       approximate objectives.
    3. Compute kernel target alignment for features from different transformations. 

* `plot.py` plots all the figures in the paper using the saved results from 
  `mnist_experiments.py`. The figures are saved in the directory `figs`.
    ```
    python plot.py
    ```
