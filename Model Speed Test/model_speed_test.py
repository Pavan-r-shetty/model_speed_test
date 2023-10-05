import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as kb
import time

# Check if GPU is available
# if not tf.config.experimental.list_physical_devices('GPU'):
    # raise RuntimeError('GPU device not found. Ensure you have TensorFlow-GPU installed.')

def custom_loss(y_actual, y_pred):
    mask = kb.greater(y_actual, 0)
    mask = tf.cast(mask, tf.float32)
    custom_loss = tf.math.reduce_sum(
        kb.square(mask*(y_actual-y_pred)))/tf.math.reduce_sum(mask)
    return custom_loss

# Load the model within the GPU device context
with tf.device('/GPU:0'):
    model = tf.keras.models.load_model('1_9_both_trt_test.h5', custom_objects={'custom_loss': custom_loss})

window_size = 80
num_channels = 8
data = np.random.rand(window_size+5,num_channels)
num_runs = 1000
times = []

# Ensure the data is placed on the GPU
with tf.device('/GPU:0'):
    x_test = tf.expand_dims(data[:,:num_channels], axis=0)
    x_test = tf.cast(x_test, dtype=tf.float32)
   
    # Warm-up
    for _ in range(10):
        _ = model.predict(x_test)

    # Time predictions
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(x_test)
        end_time = time.time()
        times.append(end_time - start_time)

average_time = sum(times) / len(times)
print(f"Average prediction time over {num_runs} runs: {average_time * 1000:.4f} milliseconds")




# trt test 
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import time

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# def load_engine(trt_runtime, engine_path):
#     """Load TensorRT engine from file."""
#     with open(engine_path, 'rb') as f:
#         engine_data = f.read()
#     return trt_runtime.deserialize_cuda_engine(engine_data)

# def allocate_buffers(engine):
#     """Allocate memory for inputs and outputs."""
#     h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
#     h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
#     d_input = cuda.mem_alloc(h_input.nbytes)
#     d_output = cuda.mem_alloc(h_output.nbytes)
#     return h_input, d_input, h_output, d_output

# def infer(context, h_input, d_input, h_output, d_output, data):
#     """Perform inference."""
#     cuda.memcpy_htod(d_input, data)
#     context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
#     cuda.memcpy_dtoh(h_output, d_output)
#     return h_output

# runtime = trt.Runtime(TRT_LOGGER)
# engine_path = 'model.trt'

# try:
#     engine = load_engine(runtime, engine_path)
#     h_input, d_input, h_output, d_output = allocate_buffers(engine)
#     context = engine.create_execution_context()

#     window_size = 80
#     num_channels = 8
#     data = np.random.rand(window_size+5, num_channels)
#     num_runs = 1000
#     times = []

#     data_prepared = data[:window_size, :num_channels].ravel().astype(np.float32)

#     # Warm-up
#     for _ in range(10):
#         infer(context, h_input, d_input, h_output, d_output, data_prepared)

#     # Time predictions
#     for _ in range(num_runs):
#         start_time = time.time()
#         _ = infer(context, h_input, d_input, h_output, d_output, data_prepared)
#         times.append(time.time() - start_time)

#     avg_time = sum(times) / len(times)
#     print(f"Average prediction time over {num_runs} runs: {avg_time * 1000:.4f} milliseconds")

# except Exception as e:
#     print(f"Error occurred: {e}")


# import numpy as np
# import tensorrt as trt
# import pycuda.driver as cuda

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# def get_engine(onnx_file_path, engine_file_path):
#     with open(onnx_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())
#     if engine is None:
#         raise ValueError("Failed to load TensorRT engine.")
#     return engine

# def allocate_buffers(engine):
#     h_input = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(engine[0].binding)), dtype=np.float32)
#     h_output = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(engine[1].binding)), dtype=np.float32)
#     d_input = cuda.mem_alloc(h_input.nbytes)
#     d_output = cuda.mem_alloc(h_output.nbytes)
#     return h_input, d_input, h_output, d_output

# def do_inference(context, h_input, d_input, h_output, d_output):
#     # Transfer input data to the GPU.
#     cuda.memcpy_htod(d_input, h_input)
#     # Execute the model.
#     context.execute_v2(bindings=[int(d_input), int(d_output)])
#     # Transfer predictions back from the GPU.
#     cuda.memcpy_dtoh(h_output, d_output)
#     return h_output

# def main():
#     onnx_file_path = 'model3.onnx'
#     engine_file_path = 'model3.trt'
    
#     engine = get_engine(onnx_file_path, engine_file_path)
#     context = engine.create_execution_context()
    
#     h_input, d_input, h_output, d_output = allocate_buffers(engine)
    
#     # Dummy data for inference.
#     h_input.fill(1)
    
#     output = do_inference(context, h_input, d_input, h_output, d_output)
#     print(output)

# if __name__ == '__main__':
#     main()
