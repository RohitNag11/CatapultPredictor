from helperlib.model_optimser import ModelOptimiser

data1_models_perf = ModelOptimiser.read_model_perf('dataset1')
data2_models_perf = ModelOptimiser.read_model_perf('dataset2')


print(data1_models_perf[: 5])
print(data2_models_perf[: 5])
