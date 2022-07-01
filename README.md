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

We will illustrate these two steps with a simple Matrix Multiplication (GEMM) example, and our desired computation will be expressed by leveraging exclusively Tensor Processing Primitives (TPP).

### Declaring the logical loops
The Matrix Multiplication algorithm multiples a Matrix A with a matrix B and gets as result an output matrix C. Matrix A is an MxK matrix (M rows and K columns), matrix B is an KxN matrix (K rows and N columns), whereas the output matrix C is an MxN matrix (M rows and N columns). The Matrix Multiplication algorithm is comprised of three logical nested loops which can be declared as follows:
```
auto gemm_loop = ThreadedLoop<3>({
     LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // a loop - Logical K loop specs
     LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // b loop - Logical M loop specs
     LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // c loop - Logical N loop specs
     loop_string);
```
Since our computation involves three logical loops we use in the declaration ```ThreadedLoop<3>```.

The first loop which has the mnemonic *a*, corresponds to a loop with start 0, upper bound Kb and step k_step, and in our use-case corresponds to the "K" loop of the GEMM (K is the inner-product/contraction dimension). This loop has an _optional_ list of step/blocking parameters {l1_k_step, l0_k_step}.

The second loop which has the mnemonic *b*, corresponds to a loop with start 0, upper bound Mb and step m_step, and in our use-case corresponds to the "M" loop of the GEMM (number of rows of the output tensor in column-major format). This loop has an _optional_ list of step/blocking parameters {l1_m_step, l0_m_step}.

The third loop which has the mnemonic *c*, corresponds to a loop with start 0, upper bound Nb and step n_step, and in our use-case corresponds to the "N" loop of the GEMM (number of columns of the output tensor in column-major format). This loop has an _optional_ list of step/blocking parameters {l1_n_step, l0_n_step}.

The specific instantiation of these loops, i.e. the loop order with which they appear, the number of times each one is blocked and also the way they are parallelized are controlled by the string *loop_string* which is provided at run-time. More specifically, the *loop_string* can be constructed using the following rules:

### RULE 1 (Loops ordering and blockings)
Each character (from *a* to *z* depending on the number of the logical loops - in our case since we have 3 logical loops the characters range from *a* to *c*) can appear in any order and any number of times. The order with which the loop characters appear in the string determine the nesting loop order, and the times each character appears determines how many times the corresponding logical loop is blocked. For example, a *loop_string* **bcabcb** corresponds to a loop where logical loop b is blocked twice (the character b appears 3 times), logical loop c is blocked once (the character c appears 2 times) and the logical loop a is not blocked (it appears only once). The blocking/tiling sizes for each logical loop level are extracted from the corresponding list of step/blocking parameters in order they appear in the list. For example, the aforementioned *loop_string* **bcabcb** correponds to the following loop nest:

```
for b0 = 0 to Mb with step l1_m_step
  for c0 = 0 to Nb with step l1_n_step
    for a0 = 0 to Kb with step k_step
      for b1 = b0 to b0 + l1_m_step with step l0_m_step
        for c1 = c0 to c0 + l1_n_step with step n_step
          for b2 = b1 to b1 + l0_m_step with step m_step
             // Logical indices to use for the computation are a0, b2, c1
```

Currently our Proof-Of-Concept (POC) implementation of PARLOOPER allows only perfectly nested blocking/tiling sizes, i.e. in the example above it should hold:
 - l1_m_step mod l0_m_step = 0
 - l0_m_step mod m_step = 0
 - l1_n_step mod n_step = 0

An important thing to note here is that all these  blocking/tiling sizes lists may be provided at runtime (e.g. one could programatically determine the blocking sizes given the problem/input at hand) and do not have to be statically determined. 

### RULE 2 (Parallelization)
If a loop character in the *loop_string* appears in its upper-case form, it dictates the intention to parallelize this loop at the specific nest-level it appears. For example, following the previous example, if the *loop_string* was **bcaBcb**, it would correspond to the following loop nest:
```
for b0 = 0 to Mb with step l1_m_step
  for c0 = 0 to Nb with step l1_n_step
    for a0 = 0 to Kb with step k_step
      #parallelize loop directive
      for b1 = b0 to b0 + l1_m_step with step l0_m_step
        for c1 = c0 to c0 + l1_n_step with step n_step
          for b2 = b1 to b1 + l0_m_step with step m_step
            // Logical indices to use for the computation are a0, b2, c1
```
Currently our POC supports 2 modes of parallelization:
1. **PAR-MODE 1: Relying on OpenMP runtime for parallelizing the loops.** By following this method, effectively the "#parallelize loop directive" above corresponds to "#pragma omp for nowait". If one wants to parallelize multiple loops, the corresponding capitalized characters should appear consecutively in the *loop_string*, and it would result in parallelization using *collapse* semantics. For example, if the *loop_string* was **bcaBCb**, PARLOOPER would yield:
```
for b0 = 0 to Mb with step l1_m_step
  for c0 = 0 to Nb with step l1_n_step
    for a0 = 0 to Kb with step k_step
      #pragma omp for collapse(2) nowait
      for b1 = b0 to b0 + l1_m_step with step l0_m_step
        for c1 = c0 to c0 + l1_n_step with step n_step
          for b2 = b1 to b1 + l0_m_step with step m_step
            // Logical indices to use for the computation are a0, b2, c1
```
When using **PAR-MODE 1** we allow to specify optionally additional directives at the end of the *loop_string* by using the special character *@* as separator. For example the loop string **bcaBCb@schedule(dynamic,1)** yields the parallelization directive:
```
      #pragma omp for collapse(2) schedule(dynamic,1) nowait
```
in the aforementioned loop nest.

The constructed loop nest is embraced by a "pragma omp parallel" region, i.e. the generated code with a *loop_string* **bcaBCb** would look like :
```
#pragma omp parallel
{
  for b0 = 0 to Mb with step l1_m_step
    for c0 = 0 to Nb with step l1_n_step
      for a0 = 0 to Kb with step k_step
        #pragma omp for collapse(2) nowait
        for b1 = b0 to b0 + l1_m_step with step l0_m_step
          for c1 = c0 to c0 + l1_n_step with step n_step
            for b2 = b1 to b1 + l0_m_step with step m_step
              // Logical indices to use for the computation are a0, b2, c1
}
```
In the user desires a barrier at the end of a specific loop-level, it may be requested using the special character "|". For example if we want to have a synchronization barrier at the end of the outermost *c* loop we can specify as *loop_string* **bc|aBCb** :
```
#pragma omp parallel
{
  for b0 = 0 to Mb with step l1_m_step {
    for c0 = 0 to Nb with step l1_n_step { 
      for a0 = 0 to Kb with step k_step {
        #pragma omp for collapse(2) nowait {
        for b1 = b0 to b0 + l1_m_step with step l0_m_step {
          for c1 = c0 to c0 + l1_n_step with step n_step {
            for b2 = b1 to b1 + l0_m_step with step m_step {
              // Logical indices to use for the computation are a0, b2, c1
            }
          }
        }
      }
    }// end of loop c0
    #pragma omp barrier
  }
}
```
2. **PAR-MODE 2: Using explicit multi-dimensional thread decompositions** Using this parallelization paradigm, the user can specify 1D, 2D or 3D loop parallelization schemes by parallelizing 1,2 or 3 loops respectively.
 * For the 1D decomposition, the threads are forming a logical "Rx1" 1D grid and are assigned the corresponding parallelized loop iterations in a block fashion. To apply this explicit decomposition, the user merely has to append after the desired upper-case loop character the substring "{R:#threads}", where #threads is a number dictating in how many ways to paralllelize that loop. For example, the *loop_string* **bC{R:16}abcb** would yield:
```
#pragma omp parallel
{
  for b0 = 0 to Mb with step l1_m_step {
    # Parallelize 16-ways loop c0
    for c0 = 0 to Nb with step l1_n_step { 
      for a0 = 0 to Kb with step k_step {
        for b1 = b0 to b0 + l1_m_step with step l0_m_step {
          for c1 = c0 to c0 + l1_n_step with step n_step {
            for b2 = b1 to b1 + l0_m_step with step m_step {
              // Logical indices to use for the computation are a0, b2, c1
            }
          }
        }
      }
    }
  }
}
```

 * For the 2D decomposition, the threads are forming a logical "RxC" 2D grid that effectively parallelizes the requested two loops. Note that the parallelized loops do **not** have to appear consecutively in the *loop_string*. For example consider the *loop_string* **bC{R:16}aB{C:4}cb**. In this example the loop *c0* is parallelized 16-ways and the loop *b1* is parallelized 4-ways using a logical thread grid of 16x4 (R=16, C=4). Each loop that is parallelized is done so in a block fashion using the requested number of "ways". Essentially PARLOOPER would yield:
```
#pragma omp parallel
{
  for b0 = 0 to Mb with step l1_m_step {
    # Parallelize 16-ways loop c0
    for c0 = 0 to Nb with step l1_n_step { 
      for a0 = 0 to Kb with step k_step {
        # Parallelize 4-ways loop b1
        for b1 = b0 to b0 + l1_m_step with step l0_m_step {
          for c1 = c0 to c0 + l1_n_step with step n_step {
            for b2 = b1 to b1 + l0_m_step with step m_step {
              // Logical indices to use for the computation are a0, b2, c1
            }
          }
        }
      }
    }
  }
}
```
 * For the 3D decomposition, the threads are forming a logical "RxCxL" 3D grid that effectively parallelizes the requested three loops. Note that the parallelized loops do **not** have to appear consecutively in the *loop_string*. For example consider the *loop_string* **bC{R:5}B{C:4}cbA{L:3}**. In this example the loop *c0* is parallelized 5-ways, the loop *b1* is parallelized 4-ways, and the loop *a0* is parallelized 3-ways using a logical thread grid of 5x4x3 (R=5, C=4, L=3). Each loop that is parallelized is done so in a block fashion using the requested number of "ways". Essentially PARLOOPER would yield:
```
#pragma omp parallel
{
  for b0 = 0 to Mb with step l1_m_step {
    # Parallelize 5-ways loop c0
    for c0 = 0 to Nb with step l1_n_step { 
      # Parallelize 4-ways loop b1
      for b1 = b0 to b0 + l1_m_step with step l0_m_step {
        for c1 = c0 to c0 + l1_n_step with step n_step {
          for b2 = b1 to b1 + l0_m_step with step m_step {
            # Parallelize 3-ways loop a0
            for a0 = 0 to Kb with step k_step {
              // Logical indices to use for the computation are a0, b2, c1
            }
          }
        }
      }
    }
  }
}
```

Now that we have described how to declare the desired nested loops along with their specification, we will show how to express the desired computation using the logical indices of the nested loops (i.e. the logical indices a0, b2, c1 in the example above). We want to emphasize that PARLOOPER will generate the desired loop nest Just-In-Time with zero lines of user-code change. The user code merely looks like
```
auto gemm_loop = ThreadedLoop<3>({
     LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // a loop - Logical K loop specs
     LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // b loop - Logical M loop specs
     LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // c loop - Logical N loop specs
     loop_string);
```
and depending on the runtime parameter *loop_string* given to PARLOOPER, arbitrarily complex parallelized loop nests will be generated Just-In-Time.

### Expressing the desired computation
Once the desired nested loop has been declared/specified, we get back an initialized *ThreadedLoop* object (i.e. the gemm_loop in the example above) which can be passed at runtime (up to) three parameters:
1. A function pointer to a function with signature:
```
void loop_body_func(int *ind)
```
This function is called at the inner-most level of the generated loop-nest, and essentially it will perform the desired computation. The function *loop_body_func* gets as input an array of integer values which contains in the first N locations the values of the logical indices used in the nested loop in alphabetical order. In essense ind[0] corresponds to the value of the logical index *a* in the current nested-loop iteration, ind[1] corresponds to the value of the logical index *b* in the current nested-loop iteration etc. This index array is automatically allocated and initialized by PARLOOPER. By leveraging the values of the logical indices the user can express the desired computation as a function of these logical indices. For convenience, we use as loop_body_function a C++ lambda expression.

2. (Optional)  A function pointer to a function with signature:
```
void init_func()
```
This function is called just before the generated loop-nest and can be used for "initialization code" purposes (e.g. code that would initialize some data structures etc). Again, for convenience we may use as init_func a C++ lambda expression.

3. (Optional)  A function pointer to a function with signature:
```
void term_func()
```
This function is called just after the generated loop-nest and can be used for "termination code" purposes (e.g. code that would clean-up some data structures etc). Again, for convenience we may use as term_func a C++ lambda expression.

Considering the GEMM example above, and using lambda expression for *loop_body_func* we can express the desired GEMM computation using TPPs (zero_tpp and brgemm_tpp) and the logical indices as follows:
```
gemm_loop(
  [&](int* ind) {
    int i_k = ind[0], i_m = ind[1], i_n = ind[2];
    unsigned long long brcount = k_step;    
    if (i_k == 0) {
      zero_tpp( &C[i_n][i_m][0][0]);
    }
    brgemm_tpp(&A[i_m][i_k][0][0], &B[i_n][i_k][0][0], &C[i_n][i_m][0][0], &brcount);
  });
```
In this example we are using blocked tensor layouts for matrices A, B and C. Putting everything together, this is how the user's code will look like for the toy GEMM example:
```
auto gemm_loop = ThreadedLoop<3>({
     LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // a loop - Logical K loop specs
     LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // b loop - Logical M loop specs
     LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // c loop - Logical N loop specs
     loop_string);
     
gemm_loop(
  [&](int* ind) {
    int i_k = ind[0], i_m = ind[1], i_n = ind[2];
    unsigned long long brcount = k_step;
    if (i_k == 0) {
      zero_tpp( &C[i_n][i_m][0][0]);
    }
    brgemm_tpp(&A[i_m][i_k][0][0], &B[i_n][i_k][0][0], &C[i_n][i_m][0][0], &brcount);
  });
```
Note that the user's code is extremely simple since it merely defines in a *declarative* way the computational loop-nest and the desired computation as a function of the 3 logical indices and TPP. At runtime, by providing proper *loop_string* parameter one could get complex parallel loop nest implementation which is JITed by PARLOOPER without any changes in the user code. For example the PARLOOPER-generated code with a *loop_string* **bcaBCb** would be equivalent to:
```
#pragma omp parallel
{
  for b0 = 0 to Mb with step l1_m_step {
    for c0 = 0 to Nb with step l1_n_step { 
      for a0 = 0 to Kb with step k_step {
        #pragma omp for collapse(2) nowait {
        for b1 = b0 to b0 + l1_m_step with step l0_m_step {
          for c1 = c0 to c0 + l1_n_step with step n_step {
            for b2 = b1 to b1 + l0_m_step with step m_step {
              // Logical indices to use for the computation are a0, b2, c1
              int i_k = a0, i_m = b2, i_n = c1;
              unsigned long long brcount = k_step;
              if (i_k == 0) {
                zero_tpp( &C[i_n][i_m][0][0]);
              }
              brgemm_tpp(&A[i_m][i_k][0][0], &B[i_n][i_k][0][0], &C[i_n][i_m][0][0], &brcount);
            }
          }
        }
      }
    }
  }
}
```
### Helper methods in PARLOOPER
The obtained *ThreadedLoop* object has a few auxiliary methods that can be used in the *loop_body_func* to express the desired computation and can be helpful/necessary e.g. when considering explicit multi-dimensional thread decompositions.

1. ``` int get_tid(int *ind)```

This method merely returns the "thread id" of the calling thread, and as such the return value ranges from 0 ... #threads-1.

2. ``` threaded_loop_par_type get_loop_par_type(char loop_name, int *ind)```

This method returns the parallelization type of the logical loop with name *loop_name*. The return type *threaded_loop_par_type* is essentially an enum:
```
typedef enum threaded_loop_par_type {
  THREADED_LOOP_NO_PARALLEL             =  0,
  THREADED_LOOP_PARALLEL_COLLAPSE       =  1,
  THREADED_LOOP_PARALLEL_THREAD_ROWS    =  2,
  THREADED_LOOP_PARALLEL_THREAD_COLS    =  3,
  THREADED_LOOP_PARALLEL_THREAD_LAYERS  =  4
} threaded_loop_par_type;
```
In the example above with *loop_string* **bcaBCb**, calling ```get_loop_par_type('a', ind) ``` would return ```THREADED_LOOP_NO_PARALLEL``` since the logical loop *a* is not parallelized whereas ```get_loop_par_type('b', ind) ``` would return ```THREADED_LOOP_PARALLEL_COLLAPSE```since the logical loop *b* has been used in a collapse-fashion parallelization scheme.

3. ``` int get_loop_parallel_degree(char loop_name, int *ind)```

This method is valid only when leveraging "PAR-MODE 2: Using explicit multi-dimensional thread decompositions" and returns the number of "ways"/degree the requested logical loop is parallelized. For example, assuming that the *loop_string* is **bC{R:5}B{C:4}cba**, then calling ```get_loop_parallel_degree('b', ind) ``` would return 4 since the logical loop *b* is parallelized 4-ways.

4. ``` int get_tid_in_parallel_dim(char loop_name, int *ind)```

When leveraging "PAR-MODE 1" this method merely returns the "thread id" of the calling thread (and as such the return value ranges from 0 ... #threads-1) **if** the corresponding logical loop *loop_name* **is parallelized**, otherwise it returns the value -1.

When leveraging "PAR-MODE 2" this method returns the "thread id" of the calling thread in the parallelized logical loop dimension. For example, assuming that the *loop_string* is **bC{R:5}B{C:4}cbA{L:3}**, then calling ```get_tid_in_parallel_dim('c', ind) ``` would return a number between 0-4 depending on the "**R**ow team" the calling thread belongs in the logical 3D thread grid of size 5x4x3 (RxCxL). In an analogous way, calling ```get_tid_in_parallel_dim('b', ind) ``` would return a number between 0-3 depending on the "**C**olumn team" this thread belongs in the logical 3D thread grid, and finally calling ```get_tid_in_parallel_dim('a', ind) ``` would return a number between 0-2 depending on the "**L**ayer team" this thread belongs in the logical 3D thread grid.

The helper methods 1-4 enable the user to express *programmatically* in the *loop_body_func* complex parallelization strategies "ahead of time" since the exact loop-nest instantiation depends on the *loop_string* which is a runtime/JIT parameter.

## Sample codes using the PARLOOPER infrastructure
For all the developed sample codes, by exporting: ```export USE_BF16=1``` during runtime, the used precision will be bfloat16, otherwise it will be single precision (float).

1. **gemm_model_fwd.cpp** : This is the full GEMM working example used in this README. It also supports chaining together multiple GEMMs to effectively implement a Multi-Layer Perceptron primitive (MLP) by setting n_layers > 1 in the argument list.
2. **gemm_model_bwd.cpp** : This is the backward-by-data pass in packpropagation of a Fully-Connected layer primitive. It supports "privately" tranposing matrix A by setting private_wt_trans = 1 in the argument list. Otherwise the matrix A is transposed upfront.
3. **gemm_model_upd.cpp** : This is the backward-by-weights pass in packpropagation of a Fully-Connected layer primitive. For bfloat16 precision it supports "privately" tranposing matrices A (vnni-formating) and B (normal transpose) by setting private_trans = 1 in the argument list, otherwise the matrices A and B are transposed upfront. This primitve also allows parallelization across the "inner-product"/contraction dimension N by setting n_partial_filters to non-zero value in the argument list (i.e. the code extracts also parallelization across the N dimension, and at the end a reduction across all partial filters is performed to calculate the final result). More specifically, if the *loop_string* has "collapse" type parallelization, then n_partial_filters should be equal to the number of threads. If the *loop_string* has "explicit" thread decomposition (PAR-MODE 2) accross the N dimension in X-ways, then n_partial_filters should have the value X.
4. **conv_model_fwd.cpp** Forward pass of a convolution layer.
5. **conv_model_bwd.cpp** Backward-by-data pass in packpropagation of a convolution layer.
6. **conv_model_upd.cpp** Backward-by-weights pass in packpropagation of a convolution layer.

## Auto-tuning codes using the PARLOOPER infrastructure
The general methodology of auto-tuning codes using the PARLOOPER infrastructure is to (auto)-generate valid *loop_string* permutations and effectively explore different parallelization strategies, loop orders, and loop blockings to find the ones that maximize performance for the platform and problem at hand. To this extend we have generated some auxiliary/helper codes:

1. **spec_loop_generator.cpp**: It can be used as a blueprint to geenrate loop permutattions.
2. **gemm_loop_tuner.sh** Can be used a bluprint to auto-tune GEMMa

## Exemplary run of sample matmul and convolution
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

