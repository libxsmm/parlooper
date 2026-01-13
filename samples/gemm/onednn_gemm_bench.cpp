/******************************************************************************
* oneDNN GEMM Benchmark Driver
* 
* Benchmark oneDNN BF16 GEMM with pack layout for weight matrix
******************************************************************************/

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <cmath>
#include <algorithm>

#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "libxsmm.h"

using namespace dnnl;

// Helper function to make the perf_event_open system call
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// Timing utilities
double getFreq() {
    return 1.0;  // Placeholder
}

double ifreq = 1.0 / getFreq();

uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

// Fill buffer with random BF16 data
void fill_random_bf16(std::vector<float>& data) {
    static std::mt19937 generator;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& d : data) {
        d = dist(generator);
    }
}

int main(int argc, char** argv) {
    // Default parameters
    long M = 1024, N = 1024, K = 1024;
    long n_sets = 1;      // Number of different A/B/C matrix sets
    long n_iters = 10;    // Number of iterations
    long check_correctness = 0;  // Correctness check flag
    
    // Parse command line arguments
    if (argc > 1) {
        N = atol(argv[1]);
    }
    if (argc > 2) {
        M = atol(argv[2]);
    }
    if (argc > 3) {
        K = atol(argv[3]);
    }
    if (argc > 4) {
        n_sets = atol(argv[4]);
        if (n_sets == -1)
        {
            double size_total = (double)2.0 * (double)1.0 * ((double)M * (double)K + (double)M * (double)N + (double)K * (double)N) / (1024.0 * 1024.0 * 1024.0);
            double low_limit_in_gb = 5.0;
            n_sets = 1;
            while (size_total < low_limit_in_gb)
            {
                n_sets++;
                size_total = (double)2.0 * (double)n_sets * ((double)M * (double)K + (double)M * (double)N + (double)K * (double)N) / (1024.0 * 1024.0 * 1024.0);
            }
            printf("Autocalculated %d layers with total size %.2g\n", n_sets, size_total);
        }
    }
    if (argc > 5) {
        n_iters = atol(argv[5]);
    }
    if (argc > 6) {
        check_correctness = atol(argv[6]);
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "oneDNN BF16 GEMM Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Matrix dimensions:" << std::endl;
    std::cout << "  A: " << M << " x " << K << " (M x K)" << std::endl;
    std::cout << "  B: " << K << " x " << N << " (K x N)" << std::endl;
    std::cout << "  C: " << M << " x " << N << " (M x N)" << std::endl;
    std::cout << "Operation: C = A * B" << std::endl;
    std::cout << "Number of sets: " << n_sets << std::endl;
    std::cout << "Number of iterations: " << n_iters << std::endl;
    std::cout << "Correctness check: " << (check_correctness ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // Create execution engine (CPU)
        dnnl::engine engine(engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);
        
        // Matrix dimensions: C (MxN) = A (MxK) * B (KxN)
        memory::dims a_dims = {M, K};
        memory::dims b_dims = {K, N};
        memory::dims c_dims = {M, N};
        
        // Allocate multiple sets of matrices
        std::vector<std::vector<float>> a_data_sets(n_sets);
        std::vector<std::vector<float>> b_data_sets(n_sets);
        std::vector<std::vector<float>> c_data_sets(n_sets);
        
        for (long set = 0; set < n_sets; set++) {
            a_data_sets[set].resize(M * K);
            b_data_sets[set].resize(K * N);
            c_data_sets[set].resize(M * N);
            
            // Initialize with random data
            fill_random_bf16(a_data_sets[set]);
            fill_random_bf16(b_data_sets[set]);
            std::fill(c_data_sets[set].begin(), c_data_sets[set].end(), 0.0f);
        }
        
        std::cout << "Allocated and initialized " << n_sets << " sets of matrices" << std::endl;
        
        // Create memory descriptors with 'any' format to let oneDNN choose optimal layout
        // For weights (B matrix), we explicitly request VNNI format for BF16
        auto a_md = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::any);
        auto b_md = memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::any);
        auto c_md = memory::desc(c_dims, memory::data_type::bf16, memory::format_tag::any);

        // Create matmul primitive descriptor
        auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md);

        std::cout << "Created matmul primitive descriptor" << std::endl;
        
        // Get the optimal memory descriptors chosen by oneDNN
        auto a_opt_md = matmul_pd.src_desc();
        auto b_opt_md = matmul_pd.weights_desc();
        auto c_opt_md = matmul_pd.dst_desc();
        
        // Print the chosen formats
        std::cout << "========================================" << std::endl;
        std::cout << "Chosen memory formats:" << std::endl;
        
        // Helper lambda to get format string with dimensions
        auto get_format_string = [](const memory::desc& md) -> std::string {
            // Get the format kind
            auto fmt = md.get_format_kind();
            auto dims = md.get_dims();
            auto padded_dims = md.get_padded_dims();
            
            std::stringstream ss;
            
            // Print logical dimensions
            ss << "logical=[";
            for (size_t i = 0; i < dims.size(); i++) {
                if (i > 0) ss << "x";
                ss << dims[i];
            }
            ss << "]";
            
            // Print padded dimensions
            ss << " padded=[";
            for (size_t i = 0; i < padded_dims.size(); i++) {
                if (i > 0) ss << "x";
                ss << padded_dims[i];
            }
            ss << "]";
            
            if (fmt == memory::format_kind::blocked) {
                auto inner_blks = md.get_inner_blks();
                auto inner_idxs = md.get_inner_idxs();
                
                // Compute total blocking factor for each dimension
                std::vector<long> blocking_factors(dims.size(), 1);
                for (size_t i = 0; i < inner_blks.size(); i++) {
                    int dim_idx = inner_idxs[i];
                    blocking_factors[dim_idx] *= inner_blks[i];
                }
                
                // Build physical dimensions: outer dims first, then all inner blocks
                std::vector<long> physical_dims;
                
                // Add outer dimensions (padded dims divided by total blocking factor)
                for (size_t i = 0; i < padded_dims.size(); i++) {
                    physical_dims.push_back(padded_dims[i] / blocking_factors[i]);
                }
                
                // Add all inner blocks in order
                for (size_t i = 0; i < inner_blks.size(); i++) {
                    physical_dims.push_back(inner_blks[i]);
                }
                
                // Print physical blocked dimensions
                ss << " physical=[";
                for (size_t i = 0; i < physical_dims.size(); i++) {
                    if (i > 0) ss << "x";
                    ss << physical_dims[i];
                }
                ss << "]";
                
                // Print block info with dimension indices
                ss << " blocks=[";
                for (size_t i = 0; i < inner_blks.size(); i++) {
                    if (i > 0) ss << "x";
                    ss << inner_blks[i] << "@dim" << inner_idxs[i];
                }
                ss << "]";
                
                // Check if it's VNNI-like (typically has a factor of 2 for BF16)
                bool has_vnni_block = false;
                for (auto blk : inner_blks) {
                    if (blk == 2) has_vnni_block = true;
                }
                if (has_vnni_block) ss << " VNNI";
            } else if (fmt == memory::format_kind::any) {
                ss << " format=any";
            } else if (fmt == memory::format_kind::undef) {
                ss << " format=undef";
            } else {
                ss << " format=plain";
            }
            
            return ss.str();
        };
        
        std::cout << "  Matrix A (src):     " << get_format_string(a_opt_md) << std::endl;
        std::cout << "  Matrix B (weights): " << get_format_string(b_opt_md) << std::endl;
        std::cout << "  Matrix C (dst):     " << get_format_string(c_opt_md) << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Create input memory descriptors in f32 format for conversion
        auto a_in_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
        auto b_in_md = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
        
        // Allocate and reorder/pack matrices for each set
        std::vector<memory> a_mems(n_sets);
        std::vector<memory> b_mems(n_sets);
        std::vector<memory> c_mems(n_sets);
        
        std::cout << "Packing/reordering matrices..." << std::endl;
        for (long set = 0; set < n_sets; set++) {
            // Create temporary f32 input memories
            auto a_in_mem = memory(a_in_md, engine);
            auto b_in_mem = memory(b_in_md, engine);
            
            // Write f32 data
            write_to_dnnl_memory(a_data_sets[set].data(), a_in_mem);
            write_to_dnnl_memory(b_data_sets[set].data(), b_in_mem);
            
            // Create bf16 optimized memories
            a_mems[set] = memory(a_opt_md, engine);
            b_mems[set] = memory(b_opt_md, engine);
            c_mems[set] = memory(c_opt_md, engine);
            
            // Reorder and convert f32 -> bf16 with optimal layout
            reorder(a_in_mem, a_mems[set]).execute(engine_stream, a_in_mem, a_mems[set]);
            reorder(b_in_mem, b_mems[set]).execute(engine_stream, b_in_mem, b_mems[set]);
            
            engine_stream.wait();
        }
        
        std::cout << "Matrices packed/reordered" << std::endl;
        
        // Correctness check if requested
        if (check_correctness) {
            std::cout << "========================================" << std::endl;
            std::cout << "Running correctness check..." << std::endl;
            
            // Allocate plain format buffers for correctness test
            std::vector<float> a_plain(M * K);
            std::vector<float> b_plain(K * N);
            std::vector<float> c_ref(M * N, 0.0f);
            std::vector<float> c_test(M * N, 0.0f);
            
            // Use first set for correctness
            a_plain = a_data_sets[0];
            b_plain = b_data_sets[0];
            
            // Reference GEMM: C = A * B using f32 with OpenMP
            #pragma omp parallel for collapse(2)
            for (long i = 0; i < M; i++) {
                for (long j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (long k = 0; k < K; k++) {
                        sum += a_plain[i * K + k] * b_plain[k * N + j];
                    }
                    c_ref[i * N + j] = sum;
                }
            }
            
            std::cout << "  Reference GEMM computed (f32)" << std::endl;
            
            // Run oneDNN GEMM with plain A/C and VNNI-packed B
            // Create plain format descriptors for A and C
            auto a_plain_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
            auto c_plain_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);
            
            auto a_plain_mem = memory(a_plain_md, engine);
            auto c_plain_mem = memory(c_plain_md, engine);
            
            write_to_dnnl_memory(a_plain.data(), a_plain_mem);
            write_to_dnnl_memory(c_test.data(), c_plain_mem);
            
            // Convert A from f32 plain to bf16 plain
            auto a_bf16_plain_md = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto a_bf16_plain_mem = memory(a_bf16_plain_md, engine);
            reorder(a_plain_mem, a_bf16_plain_mem).execute(engine_stream, a_plain_mem, a_bf16_plain_mem);
            
            // C will be bf16 plain for output
            auto c_bf16_plain_md = memory::desc(c_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto c_bf16_plain_mem = memory(c_bf16_plain_md, engine);
            
            // Create matmul primitive descriptor with plain A/C and packed B
            auto matmul_check_pd = matmul::primitive_desc(engine, a_bf16_plain_md, b_opt_md, c_bf16_plain_md);
            auto matmul_check_prim = matmul(matmul_check_pd);
            
            // Execute oneDNN GEMM
            std::unordered_map<int, memory> check_args;
            check_args.insert({DNNL_ARG_SRC, a_bf16_plain_mem});
            check_args.insert({DNNL_ARG_WEIGHTS, b_mems[0]});  // Use VNNI-packed B
            check_args.insert({DNNL_ARG_DST, c_bf16_plain_mem});
            
            matmul_check_prim.execute(engine_stream, check_args);
            engine_stream.wait();
            
            std::cout << "  oneDNN GEMM computed (bf16 with VNNI-packed B)" << std::endl;
            
            // Convert C back to f32 for comparison
            auto c_f32_out_mem = memory(c_plain_md, engine);
            reorder(c_bf16_plain_mem, c_f32_out_mem).execute(engine_stream, c_bf16_plain_mem, c_f32_out_mem);
            engine_stream.wait();
            
            read_from_dnnl_memory(c_test.data(), c_f32_out_mem);
            
            // Use libxsmm_matdiff for comparison (same as gemm_model_fwd.cpp)
            libxsmm_matdiff_info norms, diff;
            libxsmm_matdiff_clear(&norms);
            libxsmm_matdiff_clear(&diff);
            
            libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, M*N, 1, c_ref.data(), c_test.data(), 0, 0);
            
            std::cout << "==========================================\n";
            std::cout << "#           Correctness                  #\n";
            std::cout << "==========================================\n";
            std::cout << "L1 reference  : " << norms.l1_ref << std::endl;
            std::cout << "L1 test       : " << norms.l1_tst << std::endl;
            std::cout << "L2 abs.error  : " << norms.l2_abs << std::endl;
            std::cout << "L2 rel.error  : " << norms.l2_rel << std::endl;
            std::cout << "Linf abs.error: " << norms.linf_abs << std::endl;
            std::cout << "Linf rel.error: " << norms.linf_rel << std::endl;
            std::cout << "Check-norm    : " << norms.normf_rel << std::endl;
            libxsmm_matdiff_reduce(&diff, &norms);
            
            std::cout << "========================================" << std::endl;
        }
        
        // Create the matmul primitive
        auto matmul_prim = matmul(matmul_pd);
        
        std::cout << "Created matmul primitive" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Starting warmup..." << std::endl;
        
        // Warmup iterations
        for (long iter = 0; iter < 1; iter++) {
            long set = iter % n_sets;
            
            std::unordered_map<int, memory> matmul_args;
            matmul_args.insert({DNNL_ARG_SRC, a_mems[set]});
            matmul_args.insert({DNNL_ARG_WEIGHTS, b_mems[set]});
            matmul_args.insert({DNNL_ARG_DST, c_mems[set]});
            
            matmul_prim.execute(engine_stream, matmul_args);
            engine_stream.wait();
        }
        
        std::cout << "Warmup complete" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Starting benchmark..." << std::endl;
        
        // Benchmark iterations - run n_iters for each of n_sets
        // NOTE: This benchmarks similar to benchdnn's measure_perf_individual:
        // - Wait after each iteration to measure individual execution time
        // - This matches benchdnn's methodology for CPU
#if 0
        int num_cores = 64;
        printf("Using %d cores for measurement\n", num_cores);
        int *fds = (int *)malloc(sizeof(int) * num_cores);
        struct perf_event_attr pe;

        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HW_CACHE;
        pe.size = sizeof(struct perf_event_attr);
        // Use RAW type for precise L2 Miss tracking (Example for Intel 0x3F24)
        pe.type = PERF_TYPE_RAW;
        pe.config = 0x3F24; // L2_RQSTS.MISS (All L2 Misses)

        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.inherit = 1; // Ensure child threads are also counted

        // 1. Open one event per core for the current process
        for (int i = 0; i < num_cores; i++)
        {
            fds[i] = perf_event_open(&pe, 0, i, -1, 0);
            if (fds[i] < 0)
            {
                perror("perf_event_open failed");
                exit(EXIT_FAILURE);
            }
        }

        // 2. Start all counters
        for (int i = 0; i < num_cores; i++)
        {
            ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
            ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
        }
        #endif

        long total_iters = n_iters * n_sets;
        uint64_t start_tsc = rdtsc();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (long iter = 0; iter < total_iters; iter++) {
            // Cycle through matrix sets
            long set = iter % n_sets;
            
            std::unordered_map<int, memory> matmul_args;
            matmul_args.insert({DNNL_ARG_SRC, a_mems[set]});
            matmul_args.insert({DNNL_ARG_WEIGHTS, b_mems[set]});
            matmul_args.insert({DNNL_ARG_DST, c_mems[set]});
            
            matmul_prim.execute(engine_stream, matmul_args);
            // CRITICAL: Wait after each iteration to match benchdnn behavior
            // Without this, we queue all operations and only measure total batched time
            engine_stream.wait();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t end_tsc = rdtsc();

        #if 0
        // 3. Stop and sum results
        long long aggregate_l2_misses = 0;
        for (int i = 0; i < num_cores; i++)
        {
            long long count = 0;
            ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
            read(fds[i], &count, sizeof(long long));
            aggregate_l2_misses += count;
            close(fds[i]);
        }

        printf("Total L2 Misses across %d cores: %lld (Million)\n", num_cores, aggregate_l2_misses / 1000000);
        free(fds);
        #endif

        // Compute statistics
        std::chrono::duration<double> duration = end_time - start_time;
        double total_time = duration.count();
        double avg_time = total_time / total_iters;
        
        uint64_t total_cycles = end_tsc - start_tsc;
        double avg_cycles = (double)total_cycles / total_iters;
        
        double gflops = (2.0 * M * N * K * 1e-9) / avg_time;
        
        std::cout << "========================================" << std::endl;
        std::cout << "Benchmark Results" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total iterations: " << total_iters << " (" << n_iters << " per set × " << n_sets << " sets)" << std::endl;
        std::cout << "Total time: " << total_time << " seconds" << std::endl;
        std::cout << "Average time per iteration: " << avg_time * 1000.0 << " ms" << std::endl;
        std::cout << "Average cycles per iteration: " << avg_cycles << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (dnnl::error& e) {
        std::cerr << "oneDNN error: " << e.message << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
