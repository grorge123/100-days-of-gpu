# Day 002 (2026/02/05)
## What I learned
### Memory of GPU
From a software perspective, the basic unit in GPU is thread. Threads are store in a block, and each block will execute by SM. The number of threads that can be executed at one time is a multiple of the warp size.
The local variable in __global__ function is stored in thread register, and the variable with __share__ property is store in shared memory, and share with other threads in the same block. We can use cudaGetDeviceProperties to get those basic information.

### Basic cuda programing concept
We need to utilize shared memory to store the data of a function needed, since coping data from global memory is very slow, and registers are very small, it almost can not store any data. Setting The number of threads in a block is the multiple of the warp size. 

### Matrix multiplication
I use AI to generate the code to do matrix multiplication, it shows how to arrangement and positioning of threads, I will use the concept to optimize it in next chepter.