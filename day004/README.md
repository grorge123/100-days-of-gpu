# Day 004 (2026/03/23)
## What I learned
### WMMA
After talking with AI, my day003 code, AI gave me a new direction wmma. This is a new CUDA API that can use Tensor Core to do half float multiplication. Alought this new API does not accelerate the code and has some floating-point error, it provide me a new idea. I can use more high level API to do more interesting things.
Next day, I may go back to study day003's optimization, then in the future I will add the library research into my learning scope.