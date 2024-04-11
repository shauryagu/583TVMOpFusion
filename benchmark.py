import torch
import torchvision.models as models
import tvm
from tvm import relay, runtime, autotvm
from tvm.contrib.download import download_testdata
import numpy as np
import onnx
from tvm.contrib import graph_executor
import time
import pdb
import tvm.relay.testing

def get_resnet18_from_torch():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def convert_to_onnx(model, input_shape, onnx_path="./models/resnet18.onnx"):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11,
                      input_names=['input'], output_names=['output'])
    return onnx_path

def compile_model(onnx_model_path, target, shape_dict):
    onnx_model = onnx.load(onnx_model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target=target, params=params)
    return mod, params

def benchmark_model(lib, dev, input_shape, dtype='float32', iterations=100):
    module = graph_executor.GraphModule(lib['default'](dev))
    # Create random input
    data = np.random.uniform(-1, 1, size=input_shape).astype(dtype)
    module.set_input('input', data)

    # Warmup
    for _ in range(10):
        module.run()

    # Benchmark
    ftimer = module.module.time_evaluator("run", dev, number=1, repeat=iterations)
    prof_res = np.array(ftimer().results) * 1000  # Convert to milliseconds
    print(f"Average inference time (over {iterations} runs): {np.mean(prof_res):.2f} ms")
    return np.mean(prof_res), np.std(prof_res)

def prepare_dataset(input_data, input_name='input'):
    """Prepare dataset for calibration.
    
    Args:
    - input_data: NumPy array of input data for calibration.
    - input_name: Name of the input tensor expected by the model.
    
    returns:
    - iterable of dictionaries for calibration.
    """
    dataset = []
    for data_sample in input_data:
        # Convert each sample to tvm.nd.NDArray and wrap in a dictionary
        data_sample_nd = tvm.nd.array(data_sample.astype('float32'))
        dataset.append({input_name: data_sample_nd})
    return dataset

def quantize_model(mod, params, input_shape, target, dataset):
### this is the main quantized function ###

    # maybe want to experiment calibration on REAL data, see how it affects performance. 
    with relay.quantize.qconfig(calibrate_mode='kl_divergence', global_scale=8.0):
        # Annotation might be needed before this step
        quantized_mod = relay.quantize.quantize(mod, params=params, dataset=dataset)
    print(quantized_mod)
    # Compile the quantized model
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(quantized_mod, target=target, params=params)
    return lib

def bench_resnet18_tvm(lib, input_shape, target='llvm -mcpu=core-avx2', batch=1, dtype='float32', iterations=100):
    ctx = tvm.device(target, 0)
    m = graph_executor.GraphModule(lib["default"](ctx))
    adjusted_input_shape = (batch,) + input_shape[1:]
    x = np.random.uniform(size=adjusted_input_shape).astype(dtype)
    data_tvm = tvm.nd.array(x, ctx)

    # Warmup
    for _ in range(10):
        m.set_input('input', data_tvm)
        m.run()

    # Benchmark
    timer = m.module.time_evaluator("run", ctx, number=iterations)
    t = np.array(timer(data_tvm).results) * 1000  # Convert to milliseconds

    print('Quantized ResNet-18 TVM (batch={}): {:.2f} ms (mean over {} iterations)'.format(batch, t.mean(), iterations))


### BEGINING OF COMPILATION WORK FLOW ###

input_shape = (1, 3, 224, 224)
shape_dict = {'input': input_shape}
input_dtype = 'float32'
model = get_resnet18_from_torch() # get the model
onnx_model_path = convert_to_onnx(model, input_shape) # convert the model to onnx (this is just a unified ml framework)

# Compile the model in TVM
target = "llvm -mcpu=core-avx2"  # Use "cuda" for GPU
mod, params = compile_model(onnx_model_path, target, shape_dict)
dev = tvm.device(target, 0)


### NON-QUANTIZED MODEL ###
# print("Benchmarking non-quantized model...")  # UNCOMMENT if you want to benchmark tvm-compiled fp32 model
# mean_time, _ = benchmark_model(lib, dev, input_shape)
# # Average inference time (over 100 runs): 20.79 ms

### QUANTIZE MODEL ###
print("benchmarking quantized model...")
prepared_dataset = prepare_dataset(np.random.rand(1, *(1, 3, 224, 224)).astype("float32"))
quantized_lib = quantize_model(mod, params, input_shape, target, prepared_dataset)
bench_resnet18_tvm(quantized_lib, input_shape, target, batch=1, dtype='float32', iterations=100)
# # Average quantized inference time (over 100 runs): 6.70 ms