# Catapult Design Predictor

> A Neural Network model made with Python and Tensorflow Keras to predict the probability of a catapult hitting the target, given different design parameters.

## Background

Previous Imperial College London undergraduate mechanical engineering students have had a project of designing a catapult device for hitting a target object. This project has been run with many students over several years, and one
particularly thorough academic has recorded all the design parameters and how successful each design was at hitting the target.

The following parameters were recorded:

- Arm length
- Ball weight
- Ball radius
- Temperature
- Elastic constant of spring
- Weight of device

## Data Used

The performance was recorded as either hitting (1) or missing (0) the target. Data has been collected in a CSV format, with a column for each of the parameters above followed by the performance. There are two datasets, since at a point in the past the design specification was changed.
