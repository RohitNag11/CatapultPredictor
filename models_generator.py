from helperlib.model_optimser import ModelOptimiser


data1_model_optimiser = ModelOptimiser(
    'dataset1', 'data/dataset1.csv', 'Target hit', 0.3)
data2_model_optimiser = ModelOptimiser(
    'dataset2', 'data/dataset2.csv', 'Target hit', 0.3)

if __name__ == "__main__":
    data1_model_optimiser.run(5, (250, 250), (2, 4), (2, 4), [
                              'relu', 'softmax', 'tanh', 'softplus', 'sigmoid'])
    data2_model_optimiser.run(5, (250, 250), (2, 6), (2, 6), [
                              'relu', 'softmax', 'tanh', 'softplus', 'sigmoid'])
