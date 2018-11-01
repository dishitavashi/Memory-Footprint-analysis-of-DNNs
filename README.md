# Memory-Footprint-analysis-of-DNNs
Characterizing memory footprints of state-of-the-art DNNs

Memory is one of the biggest challenges in deep neural network (DNNs) today. Storing huge amounts of weights and activations in DNNs with limited memory bandwidth is difficult. Besides model size there are many other factors which will be discussed in this report that determines memory usage in a DNN. Memory characterization helps evaluate not only memory consumption but also power and speed of a particular DNN. This project comes up with a tool that allows one to evaluate memory footprint, latency and energy consumption for different state-of-the-art DNN models and memory reduction techniques and provides its memory footprint as a result.

The memory requirement of a model is affected directly by the number of parameters in the network. However, the way a network is structured is also a determining factor in memory consumption of a DNN. Other factors that determine memory usage are feature maps, gradient maps, computational techniques, batch size and workspace.

In this project, we come up with a hybrid of analytical and empirical approach where software and hardware platform dependencies are modeled empirically by running micro benchmark networks with varying parameter values like input size, channel size, weights, etc. and measuring the memory consumption for these micro benchmarks and extrapolating them to state of the art networks through analytical approach.

MICRO BENCHMARKING
As a part of first milestone, we implemented the convolution layer and fully connected layer on PyTorch and profiled the memory usage both line-by-line incrementally and overall. Further, we implemented two convolution layers back to back and two fully connected layers in the same manner on PyTorch and evaluated its memory consumption.
Memory Profiling is done using a python module for monitoring memory consumption of a process as well as line-by-line analysis of memory consumption. The line-by- line memory usage mode is used by first decorating the function we would like to profile with @profile as shown below.
