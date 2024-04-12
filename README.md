# 583TVMOpFusion
Final project for advanced compilers - memory intensive operator fusion for quantized models

Our goal is to see if the optimized operator fusion scheduling primitives for memory-intensive operators (as proposed by https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10213154) will provide a meaningful decrease in memory utilization of quantized models while maintaining accuracy. When scheduling the custom operator fusion rules, we'll have to be careful of variable types and altered data layouts that are seen in quantized models and see how the auto-scheduler adjusts loop tiling, memory caching, and operator fusion settings to find the optimal configuration for both performance and resource utilization.
