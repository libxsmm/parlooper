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
## How to use PARLOOPER
The development of applications via PARLOOPER is comprised of two steps:
1. Declaring the nested loops along with their specification
2. Expresing the desired computation using the logical indices of the nested loops.

We will illustrate these two steps with a simple Matrix Multiplication (GEMM) example, and out desired computation will be expressed be leveraging exclusively Tensor Processing Primitives (TPP).

### Declaring the logical loops
The Matrix Multiplication algorithm is comprised of three logical nested loops which can be declared as follows:
```
auto gemm_loop = ThreadedLoop<3>({
     LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // a loop - Logical K loop specs
     LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // b loop - Logical M loop specs
     LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // c loop - Logical N loop specs
     loop_string);
```
The first loop which has the mnemonic *a*, corresponds to a loop with start 0, upper bound Kb and step k_step, and for our use-csae corresponds to the "K" loop of the GEMM. This loop has an _optional_ list of step/blocking parameters {l1_k_step, l0_k_step}.

The second loop which has the mnemonic *b*, corresponds to a loop with start 0, upper bound Mb and step m_step, and for our use-csae corresponds to the "M" loop of the GEMM. This loop has an _optional_ list of step/blocking parameters {l1_m_step, l0_m_step}.

The thord loop which has the mnemonic *c*, corresponds to a loop with start 0, upper bound Nb and step N_step, and for our use-csae corresponds to the "N" loop of the GEMM. This loop has an _optional_ list of step/blocking parameters {l1_n_step, l0_n_step}.

The specific instantion of these loops, i.e. the loop order with which they appear, the number of times each one is blocked and also the way they are parallelized are controlled by the string *loop_string* provided at run-time.

The *loop_string* can be constroctued using the following rules:
1. Each character (from *a* to *z* depending on the number of the logical loops - in our case since we have 3 logical loops the characters range from *a* to *c*) can appear in any order and any number of times. The order with which the loop characters appear determine the nesting loop order, and the times each character appears determines how many times the corresponding logical loop is blocked. For example, a *loop_string* **bcabcb** correspond to a loop where logical loop b is blocked twice (it appears 3 times), logical loop c is blocked once (it appears 2 times) and the logical loop a is not blocked (it appears only once). The blocking factors for each loop (if any are requested) are extracted from the corresponding list in order they appear in the list. For example, the aforementioned string correpond to the following pseudocode

```
for b0 = 0 to Mb with step l1_m_step
  for c0 = 0 to Nb with step l1_n_step
    for a0 = 0 to Kb with step k_step
      for b1 = b0 to b0 + l1_m_step with step l0_m_step
        for c1 = c0 to c0 + l1_n_step with step n_step
          for b2 = b1 to b1 + l0_m_step with step m_step
             // Logical indices to use for the computation are a0, b2, c1
```

## Exemplary run of test matmul and forward convolution
```
salloc --nodes=1 --partition=clx --time=03:59:00
export OMP_NUM_THREADS=28
export GOMP_CPU_AFFINITY="0-27"
srun ./gemm aCBc 2048 2048 2048 32 32 32 2 1 400
srun ./conv_fwd Abcdefg 28 14 14 64 64 3 3 1 1 1 1 32 32 1 1 1 1 1 0 400
srun ./gemm_bwd bCAc 2048 4096 1024 32 32 32 4 0 400
srun ./gemm_upd cABa 2048 4096 1024 32 32 32 8 0 0 400
 ```
## Contributors
* Evangelos Georganas (Intel Corp.)
* Dhiraj Kalamkar (Intel Corp.)

