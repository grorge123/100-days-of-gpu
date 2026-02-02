# Day 001 (2026/02/02)
## What I learned
### Install nvcc
Nvcc is a compile for c++ build gpu code.
I install nvcc from nvidia website. 
In the beginning I install cuda 13.1, but I found cuda 13.1 + clangd will caused clangd has 
some error liks
```
In included file: no type named 'pointer' in 'std::_Vector_base<float, std::allocator<float>>' 
In template: no type named '_Tp_alloc_type' in 'std::_Vector_base<float, std::allocator<float>>'
```
So I downgrade the cuda to 12.8, then it successful work.

```
sudo ubuntu-drivers autoinstall
sudo reboot
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```
### Setup clangd
I am used to using clangd to help me code. But clangd give me a error message "Cannot find CUDA installation; provide its path via '--cuda-path', or pass '-nocudainc' to build without CUDA includes".
I create .clangd file under workspace, and give the compile flag. 
```
CompileFlags:
  Add:
    - --cuda-path=/etc/alternatives/cuda
```
Now we can start to write cuda code.

## VecAdd
VecAdd is a program for add two array.
In this case, I know how to write a basic function for cuda.
Next day, I will deep research the API that about CUDA.
To compile cuda code, only use `nvcc VecAdd.cu`, then a.out will generate under the directory.
