# PARLOOPER : PARallel LOOP gEneratoR
Threaded Loops Code Generation Infrastructure targeting Tensor Contraction Applications such as GEMMs, Convolutions and Fused Deep Learning Primitives

## Rationale
In our previous [work](https://arxiv.org/abs/2104.05755) we introduced the Tensor Processing Primitives (TPP), a programming abstraction striving for efficient, portable implementation of Deep Learning and High-Performance-Computing workloads with high-productivity. TPPs define a compact, yet versatile set of 2D-tensor operators (or a virtual Tensor ISA), which subsequently can be utilized as building-blocks to construct complex operators on high-dimensional tensors. 

In this programming pradigm, the user does not have to worry about low-level implementations in order to achieve bare metal performance, since this task is undertaken by the TPP backend. Instead, it is the user's responsibility to express the desired computation in terms of TPP, and also to write the required loops around the TPP primitives (henceforth called "outer loops") that essentially traverse the computation/iteration space.

These "outer loops" effectively control the parallelization scheme of the computation, and the corresponding loop orders in conjunction with potential loop blockings/tilings affect the temporal and spatial locality of the computation. Consequently, in order to get high-performance kernels the user has to write (potentially) complicated code with respect to these "outer loops" that takes into account the increasingly complex memory hierarchy of the compute platform at hand, and the increased degree of available parallelism.

PARLOOPER aims to further simplify the "outer loop" writing by enabling the user to declare the _logical_ "outer loops" along with their specifications (i.e. bounds/steps), instead of having to explicitly write the tedious loop nests involving multiple loop orders, tilings and parallelization schemes. At runtime, the user may provide a single parameter (henceforth called _loop_string_) to dictate the desired instantiation of the loop nest (i.e. loop order, loop blockings/tilings, parallelization schemes). PARLOOPER auto-generates Just-In-Time (JIT) the requested instantiation of the loop nest by taking into account the loop specifications and the single _loop_string_ runtime parameter. The resulting user's code is extremely simple, yet powerful: With zero lines of code-changes the loop nest can be instantiated to arbitrarily complex implementation that maximizes the performance on a specific platform. PARLOOPER uses internally caching schemes to avoid JIT overheads whenever possible. By leveraging PARLOOPER along with the TPP programming abstraction, the user code is simple/declarative, and naturally lends itself to trivial autotuning to explore complex/exotic outer loop instantiations with zero lines of code writing. 

## Compiler requirements
* gcc  >=  6.1.0

## Build instructions
```
bash prepare_libxsmm.sh 
make
```

## Exemplary run of test matmul and forward convolution
```
salloc --nodes=1 --partition=clx --time=03:59:00
export OMP_NUM_THREADS=28
export GOMP_CPU_AFFINITY="0-27"
srun ./gemm aCBc 2048 2048 2048 32 32 32 2 1 400
srun ./conv_fwd Abcdefg 28 14 14 64 64 3 3 1 1 1 1 32 32 1 1 1 1 1 0 400
srun ./gemm_bwd bCAc 2048 4096 1024 32 32 32 4 0 400
srun ./gemm_upd cABa 2048 4096 1024 32 32 32 8 0 400
 ```
## Contributors
* Evangelos Georganas (Intel Corp.)
* Dhiraj Kalamkar (Intel Corp.)

