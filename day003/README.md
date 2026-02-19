# Day 003 (2026/02/19)
## What I learned
### Profile
I learned nsys and ncu for profiling cuda code. You can use following command to generate report file.
```
nsys profile -o report --stats=true ./a.out
ncu -o report --kernel-name regex:gemm_rt2x2_v4_cpasync --launch-count 1 ./a.out
```
Then use the following command to open GUI.
```
nsight-sys
ncu-ui
```
Except normal profile use following command can list all section, that include other metrics on the report.
`ncu --list-sections`

### Optimize matrix multiple
I use the Profile tool and work with AI, optimize the matrix multiple code. Such that it reduce the runtime from 240 to 77.628 ms.
Next day, I will go deep into the optimization method.