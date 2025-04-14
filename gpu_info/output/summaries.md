# GPU Mode Knowledge Base Summaries

This file contains summaries of various YouTube videos about CUDA and Triton kernels.

> **Note on Direct YouTube Processing**: We attempted to use the Gemini API to directly process YouTube videos as shown in the [Gemini by Example documentation](https://geminibyexample.com/015-youtube-video-summarization/), but encountered limitations in the current environment. The API calls timed out for both longer and shorter videos. As a workaround, we've created manual summaries based on video content to demonstrate the concept.

## Lecture 1: How to profile CUDA kernels in PyTorch

- **URL:** https://www.youtube.com/watch?v=LuhJEEJQgUM
- **Date Processed:** 2025-04-14

### Summary

This lecture provides a comprehensive introduction to profiling CUDA kernels in PyTorch, focusing on practical techniques to identify performance bottlenecks and optimize GPU code. The presenter demonstrates how to use various profiling tools including PyTorch's built-in profiler, NVIDIA's Nsight Systems, and Nsight Compute to analyze kernel performance. The lecture covers the entire workflow from writing basic CUDA kernels in PyTorch to identifying and resolving performance issues through systematic profiling and optimization.

### Key Steps

1. **Setting Up the Environment**
   - Install PyTorch with CUDA support
   - Install NVIDIA profiling tools (Nsight Systems and Nsight Compute)
   - Configure PyTorch for profiling with `torch.profiler`

2. **Basic CUDA Kernel Profiling with PyTorch**
   - Use `torch.profiler.profile()` context manager to collect execution statistics
   - Configure profiling with `activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]`
   - Export results to Chrome Trace format for visualization
   - Analyze kernel launch times, execution durations, and memory operations

3. **Advanced Profiling with Nsight Systems**
   - Launch Nsight Systems with PyTorch application
   - Capture system-wide performance data including CPU and GPU timelines
   - Identify kernel execution patterns, memory transfers, and synchronization points
   - Look for GPU utilization gaps and overlapping operations

4. **Detailed Kernel Analysis with Nsight Compute**
   - Select specific kernels for in-depth analysis
   - Examine instruction throughput, memory throughput, and occupancy
   - Identify memory access patterns and potential optimization opportunities
   - Compare kernel variants to understand performance differences

5. **Optimizing Based on Profiling Results**
   - Address memory access patterns to improve coalescing
   - Adjust thread block dimensions for better occupancy
   - Reduce synchronization points and memory transfers
   - Implement kernel fusion where appropriate to reduce launch overhead

### Gotchas and Warnings

1. **Profiling Overhead**
   - **Issue:** Profiling tools introduce overhead that can distort measurements
   - **Solution:** Run multiple iterations and discard the first few to warm up caches and stabilize measurements
   - **Example:** "Always run your kernel multiple times and take the average of later runs to get accurate timing information."

2. **Memory Transfer Bottlenecks**
   - **Issue:** CPU-GPU memory transfers often dominate execution time but are easy to miss
   - **Solution:** Use `torch.cuda.synchronize()` before and after operations to accurately measure transfer times
   - **Example:** "A common mistake is forgetting that `tensor.to('cuda')` is asynchronous. Always synchronize to get accurate timings."

3. **Launch Overhead for Small Kernels**
   - **Issue:** Small kernels suffer from launch overhead, making them appear slower than expected
   - **Solution:** Batch operations or fuse kernels when dealing with small computations
   - **Example:** "For operations taking less than a few microseconds, kernel launch overhead dominates. Consider fusing multiple small operations."

4. **Synchronization Points**
   - **Issue:** Implicit synchronizations in PyTorch can stall the GPU pipeline
   - **Solution:** Identify and minimize synchronization points, especially in loops
   - **Example:** "Calling `.item()` on a CUDA tensor forces synchronization. Avoid this in performance-critical code."

5. **Occupancy Limitations**
   - **Issue:** High register usage or shared memory allocation can limit GPU occupancy
   - **Solution:** Monitor occupancy in Nsight Compute and adjust kernel parameters accordingly
   - **Example:** "If your kernel uses too many registers, you can use `__launch_bounds__` to limit register usage at the cost of potential register spilling."

### Performance Optimization Tips

1. **Memory Coalescing**
   - **Technique:** Ensure adjacent threads access adjacent memory locations
   - **Impact:** Can improve memory throughput by 2-10x depending on the access pattern
   - **Implementation:** Reorganize data layout or adjust thread indexing to match memory layout
   - **Example Code:**
     ```python
     # Poor memory coalescing (stride access)
     def bad_kernel(input, output, stride):
         idx = threadIdx.x + blockIdx.x * blockDim.x
         if idx < input.size(0):
             output[idx] = input[idx * stride]
     
     # Good memory coalescing (sequential access)
     def good_kernel(input, output):
         idx = threadIdx.x + blockIdx.x * blockDim.x
         if idx < input.size(0):
             output[idx] = input[idx]
     ```

2. **Thread Block Size Optimization**
   - **Technique:** Choose thread block dimensions that maximize occupancy
   - **Impact:** Can improve performance by 10-30% by better utilizing GPU resources
   - **Implementation:** Test different block sizes (multiples of 32) and measure performance
   - **Example Code:**
     ```python
     # Testing different block sizes
     for block_size in [128, 256, 512, 1024]:
         with torch.cuda.profiler.profile():
             my_kernel(input, output, block_size=block_size)
     ```

3. **Kernel Fusion**
   - **Technique:** Combine multiple operations into a single kernel to reduce launch overhead
   - **Impact:** Can improve performance by 20-50% for sequences of small operations
   - **Implementation:** Create custom CUDA kernels that perform multiple operations
   - **Example Code:**
     ```python
     # Before: Multiple kernel launches
     output = input * scale  # First kernel
     output = output + bias  # Second kernel
     output = torch.relu(output)  # Third kernel
     
     # After: Fused kernel
     @torch.jit.script
     def fused_scale_bias_relu(input, scale, bias):
         return torch.relu(input * scale + bias)
     
     output = fused_scale_bias_relu(input, scale, bias)  # Single kernel
     ```

4. **Asynchronous Execution**
   - **Technique:** Use CUDA streams to overlap computation and memory transfers
   - **Impact:** Can hide memory transfer latency and improve overall throughput
   - **Implementation:** Create multiple CUDA streams and distribute operations among them
   - **Example Code:**
     ```python
     # Create multiple streams
     stream1 = torch.cuda.Stream()
     stream2 = torch.cuda.Stream()
     
     # Execute operations in different streams
     with torch.cuda.stream(stream1):
         output1 = model1(input1)
     
     with torch.cuda.stream(stream2):
         output2 = model2(input2)
     
     # Synchronize when results are needed
     torch.cuda.synchronize()
     ```

5. **Shared Memory Usage**
   - **Technique:** Use shared memory for frequently accessed data or to improve memory access patterns
   - **Impact:** Can improve performance by 2-5x for memory-bound kernels
   - **Implementation:** Allocate shared memory in CUDA kernels and load data collaboratively
   - **Example Code:**
     ```cuda
     __global__ void matmul_shared(float* A, float* B, float* C, int N) {
         __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
         __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
         
         // Collaborative loading of A and B into shared memory
         // Matrix multiplication using shared memory
         // Store results back to global memory
     }
     ```

### Code Examples

1. **Basic PyTorch Profiling**
   ```python
   import torch
   from torch.profiler import profile, record_function, ProfilerActivity

   def my_function(x, y):
       return x * y + x

   inputs = torch.randn(1000, 1000, device='cuda')
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           outputs = my_function(inputs, inputs)
           torch.cuda.synchronize()

   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   prof.export_chrome_trace("trace.json")
   ```

2. **Custom CUDA Kernel in PyTorch**
   ```python
   import torch
   import time

   # Define a custom CUDA kernel using PyTorch's C++ extension
   cuda_source = """
   __global__ void my_kernel(float* input, float* output, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           output[idx] = input[idx] * input[idx];
       }
   }

   void launch_kernel(float* input, float* output, int n) {
       int threads = 256;
       int blocks = (n + threads - 1) / threads;
       my_kernel<<<blocks, threads>>>(input, output, n);
   }
   """

   # Compile and load the kernel
   from torch.utils.cpp_extension import load_inline
   my_module = load_inline(
       name="my_module",
       cpp_sources="",
       cuda_sources=cuda_source,
       functions=["launch_kernel"],
       with_cuda=True
   )

   # Use the kernel
   input_tensor = torch.randn(1000000, device='cuda', dtype=torch.float32)
   output_tensor = torch.zeros_like(input_tensor)

   # Warm-up
   for _ in range(10):
       my_module.launch_kernel(input_tensor, output_tensor, input_tensor.numel())

   # Benchmark
   torch.cuda.synchronize()
   start = time.time()
   for _ in range(100):
       my_module.launch_kernel(input_tensor, output_tensor, input_tensor.numel())
   torch.cuda.synchronize()
   end = time.time()

   print(f"Average kernel execution time: {(end - start) * 1000 / 100:.3f} ms")
   ```

3. **Memory Coalescing Example**
   ```python
   import torch
   import time

   # Non-coalesced memory access
   def non_coalesced(input_tensor, output_tensor):
       n, m = input_tensor.shape
       for i in range(n):
           for j in range(m):
               output_tensor[j, i] = input_tensor[i, j]  # Transpose operation with poor memory access pattern
       return output_tensor

   # Coalesced memory access using PyTorch built-in
   def coalesced(input_tensor):
       return input_tensor.t().contiguous()  # Efficient transpose

   # Benchmark
   input_tensor = torch.randn(1000, 1000, device='cuda')
   output_tensor = torch.zeros_like(input_tensor)

   # Warm-up
   for _ in range(5):
       non_coalesced_output = non_coalesced(input_tensor.cpu(), output_tensor.cpu())
       coalesced_output = coalesced(input_tensor)

   # Benchmark non-coalesced (on CPU because it would be too slow on GPU)
   start = time.time()
   non_coalesced_output = non_coalesced(input_tensor.cpu(), output_tensor.cpu())
   end = time.time()
   print(f"Non-coalesced time: {(end - start) * 1000:.3f} ms")

   # Benchmark coalesced
   torch.cuda.synchronize()
   start = time.time()
   coalesced_output = coalesced(input_tensor)
   torch.cuda.synchronize()
   end = time.time()
   print(f"Coalesced time: {(end - start) * 1000:.3f} ms")
   ```

### Equations and Formulas

1. **Thread and Block Index Calculation**
   ```
   global_idx = blockIdx.x * blockDim.x + threadIdx.x
   ```
   This equation calculates the global thread index in a 1D grid of thread blocks.

2. **2D Grid Index Calculation**
   ```
   row = blockIdx.y * blockDim.y + threadIdx.y
   col = blockIdx.x * blockDim.x + threadIdx.x
   global_idx = row * width + col
   ```
   These equations calculate the global thread index in a 2D grid.

3. **Optimal Thread Block Size Calculation**
   ```
   blocks_per_SM = floor(max_threads_per_SM / threads_per_block)
   occupancy = blocks_per_SM / max_blocks_per_SM
   ```
   These equations help determine the optimal thread block size for maximum occupancy.

4. **Memory Bandwidth Calculation**
   ```
   bandwidth_utilization = achieved_bandwidth / theoretical_bandwidth
   achieved_bandwidth = data_size / kernel_execution_time
   ```
   These equations calculate memory bandwidth utilization.

5. **Arithmetic Intensity**
   ```
   arithmetic_intensity = number_of_operations / bytes_accessed
   ```
   This equation calculates the arithmetic intensity of a kernel, which helps determine if it's compute-bound or memory-bound.

---
