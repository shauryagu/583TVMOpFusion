import torch
import torchvision.models as models
import tvm
from tvm import relay, runtime, autotvm, te, auto_scheduler, topi
from tvm.contrib.download import download_testdata
import numpy as np
import onnx
from tvm.contrib import graph_executor
import time
import pdb
import tvm.relay.testing
from tvm.relay.analysis.operations_distribution import analyze_operations_distribution
# run these three commands whenever you open this file for the first time:
# source finalproj/bin/activate
# export TVM_HOME=/n/eecs583a/home/shagund/tvm
# export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

class OpExtractor(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.operations = set()

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            self.operations.add(call.op.name)
        super().visit_call(call)
        
class PatternFinder(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.patterns = []

    def visit_call(self, call):
        if call.op.name == 'nn.global_avg_pool2d':
            if isinstance(call.args[0], relay.Call) and call.args[0].op.name == 'multiply':
                self.patterns.append((call.args[0], call))
        super().visit_call(call)

def get_efficientNet_from_torch():
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

def convert_to_onnx(model, input_shape, onnx_path="./models/effNet.onnx"):
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
    # print(quantized_mod)
    # extractor = OpExtractor()
    # extractor.visit_call(mod['main'].body)
    # print("operations in eff net: ", extractor.operations)
    finder = PatternFinder()
    finder.visit(mod["main"].body)
    print(f"found {len(finder.patterns)} patterns")
    # Compile the quantized model
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(quantized_mod, target=target, params=params)
    return lib

def bench_resnet18_tvm(lib, input_shape, target='llvm -mcpu=core-avx2', batch=1, dtype='float32', iterations=100):
    ctx = tvm.device(target, 0)
    m = graph_executor.GraphModule(lib["default"](ctx))
    # relay_graph = m.get_func('main')
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

    print('Quantized EffNet TVM (batch={}): {:.2f} ms (mean over {} iterations)'.format(batch, t.mean(), iterations))

### THIS FUNCTION  IS THE TE COMPUTATION DEF ###
@auto_scheduler.register_workload
def reduce_broadcast_multiply_workload(data_shape, scale_shape):
    data = te.placeholder(data_shape, name="data")
    scale = te.placeholder(scale_shape, name="scale")
    
    reduced = topi.nn.global_avg_pool2d(data)
    multiplied = topi.multiply(reduced, scale)
    
    return [data, scale, multiplied]

# this function is based on the custom scheduling algorithm in the paper
def custom_scheduling_rule(attrs, outs, target):
    # Get the output tensor
    output = outs[0]
    # Get the input tensors
    data, scale = output.op.input_tensors
    # Get the global average pooling 2D and multiply operations
    reduced = output.op.input_tensors[0]
    multiplied = output
    # Check if the pattern matches Reduce+Broadcast/ElemWise
    if isinstance(reduced.op, tvm.relay.op.op.comm_reduce) and isinstance(multiplied.op, (tvm.relay.op.op.broadcast, tvm.relay.op.op.elemwise)):
        # Check if the input tensor dimension is greater than 1
        if len(data.shape) > 1:
            # Create a schedule
            s = te.create_schedule([output.op])
            # Creation of the local register variable
            # (Not applicable in this case, as we are using existing TVM operators)
            # Partition the Oidx operator into two dimensions using split and follow_split scheduling primitives
            num_spatial_dims = len(data.shape) - 2
            block_axis = s[reduced].op.reduce_axis[0]
            if num_spatial_dims > 1:
                outer_axes = s[reduced].op.reduce_axis[1:]
                s[reduced].reorder(block_axis, *outer_axes)
                fused_outer_axes = s[reduced].fuse(*outer_axes)
                s[reduced].parallel(fused_outer_axes)
            else:
                s[reduced].parallel(block_axis)
            # Fuse Oidx and Oidx+1 operators to the innermost parallelizable loop layer using the compute_at primitive
            s[multiplied].compute_at(s[reduced], block_axis)
            return s
    # If the pattern doesn't match or input tensor dimension is 1, return an empty schedule
    return te.create_schedule([output.op])

def optimize_reduce_broadcast_multiply(data_shape, scale_shape, target):
    # Create the search policy with the custom scheduling rule
    search_policy = auto_scheduler.SearchPolicy(custom_scheduling_rule)
    task = auto_scheduler.SearchTask(func=reduce_broadcast_multiply_workload, args=(data_shape, scale_shape), target=target)
    # Create the tuner with the search policy
    tuner = auto_scheduler.TaskScheduler([task], search_policy=search_policy)
    # Run the search
    tuner.tune()
    # get the best schedule - probably want to output this to a log?
    best_schedule = tuner.best_schedule
    optimized_func = tvm.build(best_schedule, target=target)
    
    return optimized_func

### BEGINING OF COMPILATION WORK FLOW ###

input_shape = (1, 3, 224, 224)
shape_dict = {'input': input_shape}
input_dtype = 'float32'
model = get_efficientNet_from_torch() # get the model

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