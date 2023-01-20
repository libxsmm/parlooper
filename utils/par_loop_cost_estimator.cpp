/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "par_loop_cost_estimator.h"
#include <ctype.h>
#include <dlfcn.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

void set_platform_specs(
    platform_type platform,
    int n_threads,
    platform_spec_t* platform_specs) {
  if (platform == CLX) {
    double total_threads = 28.0;
    double load_factor_to_spill_cache = 0.75;
    platform_specs->l1_size_in_bytes =
        32.0 * 1024.0 * load_factor_to_spill_cache;
    platform_specs->l2_size_in_bytes =
        1.0 * 1024.0 * 1024.0 * load_factor_to_spill_cache;
    platform_specs->llc_size_in_bytes = 38.5 * 1024.0 * 1024.0 /
        (total_threads / (1.0 * n_threads)) * load_factor_to_spill_cache;
    platform_specs->l1_bytes_per_cycle = 100.0;
    platform_specs->l2_bytes_per_cycle = 50.0;
    platform_specs->llc_bytes_per_cycle = 8.0;
    platform_specs->mem_bytes_per_cycle = 2.0;
    platform_specs->bf16_flops_per_cycle_out_of_L2 = 50.0;
    platform_specs->fp32_flops_per_cycle_out_of_L2 = 57.0;
    platform_specs->freq_in_ghz = 1.8;
    platform_specs->mem_hierarchy = MEM_LLC_L2_L1;
    platform_specs->n_threads = n_threads;
  }
  return;
}

void set_tensor_metadata(
    int M,
    int N,
    int K,
    int brcount,
    int dtype_size,
    tensor_metadata_t* tensor_metadata) {
  tensor_metadata->M = M;
  tensor_metadata->N = N;
  tensor_metadata->K = K;
  tensor_metadata->brcount = brcount;
  tensor_metadata->dtype_size = dtype_size;
  tensor_metadata->subtensor_a_size_bytes = M * K * brcount * dtype_size;
  tensor_metadata->subtensor_b_size_bytes = N * K * brcount * dtype_size;
  tensor_metadata->subtensor_c_size_bytes = M * N * dtype_size;
  return;
}

void count_distinct_accesses(
    std::vector<std::string>& trace,
    int start,
    int end,
    int* a_distinct_subtensors,
    int* b_distinct_subtensors,
    int* c_distinct_subtensors) {
  std::unordered_map<std::string, int> counterMap;
  int i, a_distinct = 0, b_distinct = 0, c_distinct = 0;
  for (i = start; i <= end; i++) {
    /* We check if we have encountered the entry in the given timeframe...*/
    if (counterMap.find(trace[i]) == counterMap.end()) {
      counterMap.insert({trace[i], 0});
      if (i % 3 == 0) {
        a_distinct++;
      } else if (i % 3 == 1) {
        b_distinct++;
      } else {
        c_distinct++;
      }
    }
  }
  *a_distinct_subtensors = a_distinct;
  *b_distinct_subtensors = b_distinct;
  *c_distinct_subtensors = c_distinct;
  return;
}

void update_access_type_info(
    platform_spec_t platform_spec,
    int intermediate_data_size,
    int subtensor_size,
    mem_hierarchy_loc* resident_location,
    data_volume_t* data_vol) {
  if (platform_spec.mem_hierarchy == MEM_LLC_L2_L1) {
    if (intermediate_data_size < platform_spec.l1_size_in_bytes) {
      data_vol->l1_bytes += subtensor_size;
      *resident_location = L1;
    } else if (intermediate_data_size < platform_spec.l2_size_in_bytes) {
      data_vol->l2_bytes += subtensor_size;
      *resident_location = L2;
    } else if (intermediate_data_size < platform_spec.llc_size_in_bytes) {
      data_vol->llc_bytes += subtensor_size;
      *resident_location = LLC;
    } else {
      data_vol->mem_bytes += subtensor_size;
      *resident_location = MEM;
    }
  }
  if (platform_spec.mem_hierarchy == MEM_L2_L1) {
    if (intermediate_data_size < platform_spec.l1_size_in_bytes) {
      data_vol->l1_bytes += subtensor_size;
      *resident_location = L1;
    } else if (intermediate_data_size < platform_spec.l2_size_in_bytes) {
      data_vol->l2_bytes += subtensor_size;
      *resident_location = L2;
    } else {
      data_vol->mem_bytes += subtensor_size;
      *resident_location = MEM;
    }
  }
  return;
}

#if 1
double cycles_for_brgemm(
    tensor_metadata_t tensor_metadata,
    mem_hierarchy_loc cur_a_loc,
    mem_hierarchy_loc cur_b_loc,
    mem_hierarchy_loc cur_c_loc,
    platform_spec_t platform_spec) {
  double result = 0.0;
  double flops_per_brgemm = 2 * tensor_metadata.M * tensor_metadata.N *
      tensor_metadata.K * tensor_metadata.brcount;
  double flops_per_cycle_out_of_L2 = (tensor_metadata.dtype_size == 2) ? platform_spec.bf16_flops_per_cycle_out_of_L2 : platform_spec.fp32_flops_per_cycle_out_of_L2;
  double compute_cycles_out_of_L2 = flops_per_brgemm / flops_per_cycle_out_of_L2;
  double llc_bytes_per_cycle = platform_spec.llc_bytes_per_cycle;
  double mem_bytes_per_cycle = platform_spec.mem_bytes_per_cycle;
  /* We assume that the L2-resident GEMM microkernel is becnhmarked at the
   * granularity of 32x32 */
  int m_blocks = (tensor_metadata.M + 31) / 32;
  int n_blocks = (tensor_metadata.N + 31) / 32;
  int gemm_microkernels = m_blocks * n_blocks;
  int l2_resident_gemm_microkernels = 0;

  if ((cur_a_loc == L1 || cur_a_loc == L2) &&
      (cur_b_loc == L1 || cur_b_loc == L2)) {
    result = compute_cycles_out_of_L2;
  }

  if ((cur_a_loc == L1 || cur_a_loc == L2) &&
      (cur_b_loc == LLC || cur_b_loc == MEM)) {
    /* B comes from LLC/MEM as such #n_blocks GEMMs are bound by data movement
     */
    l2_resident_gemm_microkernels = (m_blocks - 1) * n_blocks;
    if (cur_b_loc == LLC) {
      result = tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    } else {
      result = tensor_metadata.subtensor_b_size_bytes / mem_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    }
  }

  if ((cur_b_loc == L1 || cur_b_loc == L2) &&
      (cur_a_loc == LLC || cur_a_loc == MEM)) {
    /* A comes from LLC/MEM as such #m_blocks GEMMs are bound by data movement
     */
    l2_resident_gemm_microkernels = (n_blocks - 1) * m_blocks;
    if (cur_a_loc == LLC) {
      result = tensor_metadata.subtensor_a_size_bytes / llc_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    } else {
      result = tensor_metadata.subtensor_a_size_bytes / mem_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    }
  }

  if ((cur_b_loc == LLC || cur_b_loc == MEM) &&
      (cur_a_loc == LLC || cur_a_loc == MEM)) {
    /* A and B come from LLC/MEM as such (#m_blocks+#n_blocks-1) GEMMs are bound
     * by data movement */
    l2_resident_gemm_microkernels =
        n_blocks * m_blocks - (m_blocks + n_blocks - 1);
    if (cur_a_loc == LLC) {
      if (cur_b_loc == LLC) {
        result = tensor_metadata.subtensor_a_size_bytes / llc_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      } else {
        result = tensor_metadata.subtensor_a_size_bytes / llc_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / mem_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      }
    } else {
      if (cur_b_loc == LLC) {
        result = tensor_metadata.subtensor_a_size_bytes / mem_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      } else {
        result = tensor_metadata.subtensor_a_size_bytes / mem_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      }
    }
  }

  if (cur_c_loc == LLC || cur_c_loc == MEM) {
    double c_move_cycles = 0.0;
    if (cur_c_loc == LLC) {
      c_move_cycles =
          tensor_metadata.subtensor_c_size_bytes / llc_bytes_per_cycle;
    } else {
      c_move_cycles =
          tensor_metadata.subtensor_c_size_bytes / mem_bytes_per_cycle;
    }
    result += c_move_cycles;
  }

  return result;
}
#else
double cycles_for_brgemm(
    tensor_metadata_t tensor_metadata,
    mem_hierarchy_loc cur_a_loc,
    mem_hierarchy_loc cur_b_loc,
    platform_spec_t platform_spec) {
  double result = 0.0;
  double flops_per_brgemm = 2 * tensor_metadata.M * tensor_metadata.N *
      tensor_metadata.K * tensor_metadata.brcount /*/ 1000000000.0*/;
  double compute_cycles_out_of_L2 =
      flops_per_brgemm / platform_spec.flops_per_cycle_out_of_L2;
  double llc_bytes_per_cycle = platform_spec.llc_bytes_per_cycle;
  double mem_bytes_per_cycle = platform_spec.mem_bytes_per_cycle;
  double gflops_l2 = 1064.0;
  double gflops_llc = 735.0;
  /* We assume that the L2-resident GEMM microkernel is becnhmarked at the
   * granularity of 32x32 */
  int m_blocks = (tensor_metadata.M + 31) / 32;
  int n_blocks = (tensor_metadata.N + 31) / 32;
  int gemm_microkernels = m_blocks * n_blocks;
  int l2_resident_gemm_microkernels = 0;

  if ((cur_a_loc == L1 || cur_a_loc == L2) &&
      (cur_b_loc == L1 || cur_b_loc == L2)) {
    // result = compute_cycles_out_of_L2;
    result = (flops_per_brgemm / gflops_l2) * platform_spec.freq_in_ghz;
  }

  if ((cur_a_loc == L1 || cur_a_loc == L2) &&
      (cur_b_loc == LLC || cur_b_loc == MEM)) {
    /* B comes from LLC/MEM as such #n_blocks GEMMs are bound by data movement
     */
    l2_resident_gemm_microkernels = (m_blocks - 1) * n_blocks;
    if (cur_b_loc == LLC) {
      result = tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    } else {
      result = tensor_metadata.subtensor_b_size_bytes / mem_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    }
    result = (flops_per_brgemm / gflops_llc) * platform_spec.freq_in_ghz;
  }

  if ((cur_b_loc == L1 || cur_b_loc == L2) &&
      (cur_a_loc == LLC || cur_a_loc == MEM)) {
    /* A comes from LLC/MEM as such #m_blocks GEMMs are bound by data movement
     */
    l2_resident_gemm_microkernels = (n_blocks - 1) * m_blocks;
    if (cur_a_loc == LLC) {
      result = tensor_metadata.subtensor_a_size_bytes / llc_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    } else {
      result = tensor_metadata.subtensor_a_size_bytes / mem_bytes_per_cycle +
          (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
              gemm_microkernels;
    }
    result = (flops_per_brgemm / gflops_llc) * platform_spec.freq_in_ghz;
  }

  if ((cur_b_loc == LLC || cur_b_loc == MEM) &&
      (cur_a_loc == LLC || cur_a_loc == MEM)) {
    /* A and B come from LLC/MEM as such (#m_blocks+#n_blocks-1) GEMMs are bound
     * by data movement */
    l2_resident_gemm_microkernels =
        n_blocks * m_blocks - (m_blocks + n_blocks - 1);
    if (cur_a_loc == LLC) {
      if (cur_b_loc == LLC) {
        result = tensor_metadata.subtensor_a_size_bytes / llc_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      } else {
        result = tensor_metadata.subtensor_a_size_bytes / llc_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / mem_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      }
    } else {
      if (cur_b_loc == LLC) {
        result = tensor_metadata.subtensor_a_size_bytes / mem_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      } else {
        result = tensor_metadata.subtensor_a_size_bytes / mem_bytes_per_cycle +
            tensor_metadata.subtensor_b_size_bytes / llc_bytes_per_cycle +
            (compute_cycles_out_of_L2 * l2_resident_gemm_microkernels) /
                gemm_microkernels;
      }
    }
    // result = (flops_per_brgemm / gflops_llc)*platform_spec.freq_in_ghz;
  }

  return result;
}

#endif

double tensor_contraction_cost_estimator(
    cost_analysis_type analysis_type,
    std::vector<std::string>* traces_array,
    tensor_metadata_t tensor_metadata,
    platform_spec_t platform_spec) {
  int threads_to_analyze, t;
  double total_times[128];
  double max_time = 0.0;
  double min_time = 0.0;
  memset(total_times, 0, sizeof(double));

  if (analysis_type == SINGLE_TRACE) {
    threads_to_analyze = 1;
  }
  if (analysis_type == PARALLEL_TRACES) {
    threads_to_analyze = platform_spec.n_threads;
  }

#pragma omp parallel for
  for (int _t = 0; _t < threads_to_analyze; _t++) {
    std::vector<int> cur_val{-1, 0};
    int i, prev, cur_count, total_distance = 0, itm_data_size = 0,
                            a_distinct = 0, b_distinct = 0, c_distinct = 0;
    data_volume_t a_data_vol, b_data_vol, c_data_vol;
    mem_hierarchy_loc cur_a_loc, cur_b_loc, cur_c_loc;
    int subtensor_a_size_bytes = tensor_metadata.subtensor_a_size_bytes;
    int subtensor_b_size_bytes = tensor_metadata.subtensor_b_size_bytes;
    int subtensor_c_size_bytes = tensor_metadata.subtensor_c_size_bytes;
    double total_cycles = 0.0;
    std::vector<std::string>& trace = traces_array[_t];
    std::unordered_map<std::string, std::vector<int>> tensorAccessMap;

    memset(&a_data_vol, 0, sizeof(data_volume_t));
    memset(&b_data_vol, 0, sizeof(data_volume_t));
    memset(&c_data_vol, 0, sizeof(data_volume_t));

    /* Initialize hash map with trace entries */
    for (i = 0; i < trace.size(); i++) {
      tensorAccessMap.insert({trace[i], cur_val});
    }
    for (i = 0; i < trace.size(); i++) {
      auto it = tensorAccessMap.find(trace[i]);
      prev = it->second[0];
      /* Negative prev value means it is the first access of subtensor within
       * this trace, otherwise it has been seen before */
      if (prev < 0) {
        if (i % 3 == 0) {
          a_data_vol.mem_bytes += subtensor_a_size_bytes;
          cur_a_loc = MEM;
        } else if (i % 3 == 1) {
          b_data_vol.mem_bytes += subtensor_b_size_bytes;
          cur_b_loc = MEM;
        } else {
          c_data_vol.mem_bytes += subtensor_c_size_bytes;
          cur_c_loc = MEM;
        }
        it->second[0] = i;
      } else {
        if (i % 3 == 0) {
          /* This is an A access */
          count_distinct_accesses(
              trace, prev, i - 1, &a_distinct, &b_distinct, &c_distinct);
          a_distinct--;
          itm_data_size = a_distinct * subtensor_a_size_bytes +
              b_distinct * subtensor_b_size_bytes +
              c_distinct * subtensor_c_size_bytes;
          update_access_type_info(
              platform_spec,
              itm_data_size,
              subtensor_a_size_bytes,
              &cur_a_loc,
              &a_data_vol);
        } else if (i % 3 == 1) {
          /* This is a B access */
          count_distinct_accesses(
              trace, prev - 1, i - 2, &a_distinct, &b_distinct, &c_distinct);
          b_distinct--;
          itm_data_size = a_distinct * subtensor_a_size_bytes +
              b_distinct * subtensor_b_size_bytes +
              c_distinct * subtensor_c_size_bytes;
          update_access_type_info(
              platform_spec,
              itm_data_size,
              subtensor_b_size_bytes,
              &cur_b_loc,
              &b_data_vol);
        } else {
          /* This is a C access */
          count_distinct_accesses(
              trace, prev - 2, i - 3, &a_distinct, &b_distinct, &c_distinct);
          c_distinct--;
          itm_data_size = a_distinct * subtensor_a_size_bytes +
              b_distinct * subtensor_b_size_bytes +
              c_distinct * subtensor_c_size_bytes;
          update_access_type_info(
              platform_spec,
              itm_data_size,
              subtensor_c_size_bytes,
              &cur_c_loc,
              &c_data_vol);
        }
        /* Update new prev location and overall reuse distance for this slice */
        it->second[0] = i;
        // it->second[1] += a_distinct + b_distinct;
      }

      if (i % 3 == 2) {
        total_cycles += cycles_for_brgemm(
            tensor_metadata, cur_a_loc, cur_b_loc, cur_c_loc, platform_spec);
      }
    }
    total_times[_t] = total_cycles / platform_spec.freq_in_ghz * (0.000001);
    // printf("Total milliseconds: %5g\n",
    // total_cycles/platform_spec.freq_in_ghz*(0.000001));

#if 1
    // int tid = omp_get_thread_num();
    if (0) {
      printf(
          "Total A MBytes:\nDRAM\t\tLLC\t\tL2\t\tL1\n%.3g\t\t%.3g\t\t%.3g\t\t%.3g\n",
          a_data_vol.mem_bytes / 1024.0 / 1024.0,
          a_data_vol.llc_bytes / 1024.0 / 1024.0,
          a_data_vol.l2_bytes / 1024.0 / 1024.0,
          a_data_vol.l1_bytes / 1024.0 / 1024.0);

      printf(
          "Total B MBytes:\nDRAM\t\tLLC\t\tL2\t\tL1\n%.3g\t\t%.3g\t\t%.3g\t\t%.3g\n",
          b_data_vol.mem_bytes / 1024.0 / 1024.0,
          b_data_vol.llc_bytes / 1024.0 / 1024.0,
          b_data_vol.l2_bytes / 1024.0 / 1024.0,
          b_data_vol.l1_bytes / 1024.0 / 1024.0);

      printf(
          "Total C MBytes:\nDRAM\t\tLLC\t\tL2\t\tL1\n%.3g\t\t%.3g\t\t%.3g\t\t%.3g\n",
          c_data_vol.mem_bytes / 1024.0 / 1024.0,
          c_data_vol.llc_bytes / 1024.0 / 1024.0,
          c_data_vol.l2_bytes / 1024.0 / 1024.0,
          c_data_vol.l1_bytes / 1024.0 / 1024.0);
    }
#endif
  }

  max_time = total_times[0];
  min_time = total_times[0];
  for (t = 1; t < threads_to_analyze; t++) {
    if (total_times[t] > max_time) {
      max_time = total_times[t];
    }
    if (total_times[t] < min_time) {
      min_time = total_times[t];
    }
  }
  // printf("MAX time is %.5g and MIN time is %.5g\n", max_time, min_time);

  return max_time;
}

#if 0
int main(int argc, char** argv) {
  platform_spec_t my_platform;
  int subtensor_size_bytes = 32 * 1024 * 2;
  my_platform.l1_size_in_bytes = 48 * 1024;
  my_platform.l2_size_in_bytes = 2 * 1024 * 1024;
  my_platform.llc_size_in_bytes = 105 * 1024 * 1024;
  my_platform.l1_bytes_per_cycle = 128.0;
  my_platform.l2_bytes_per_cycle = 64.0;
  my_platform.llc_bytes_per_cycle = 8.0;
  my_platform.dram_bytes_per_cycle = 2.2;

  std::vector<std::string> inp_trace = {"A[0][0]",
                                        "B[0][0]",
                                        "A[0][0]",
                                        "B[0][1]",
                                        "A[0][1]",
                                        "B[0][0]",
                                        "A[0][1]",
                                        "B[0][1]",
                                        "A[0][2]",
                                        "B[0][0]",
                                        "A[0][2]",
                                        "B[0][1]",
                                        "A[0][3]",
                                        "B[0][0]",
                                        "A[0][3]",
                                        "B[0][1]"};

  std::vector<std::string> inp_trace2 = {"A[0][0]",
                                         "B[0][0]",
                                         "A[0][1]",
                                         "B[0][0]",
                                         "A[0][2]",
                                         "B[0][0]",
                                         "A[0][3]",
                                         "B[0][0]",
                                         "A[0][0]",
                                         "B[0][1]",
                                         "A[0][1]",
                                         "B[0][1]",
                                         "A[0][2]",
                                         "B[0][1]",
                                         "A[0][3]",
                                         "B[0][1]"};

  tensor_contraction_cost_estimator(inp_trace, subtensor_size_bytes, subtensor_size_bytes, my_platform);

  tensor_contraction_cost_estimator(inp_trace2, subtensor_size_bytes, subtensor_size_bytes, my_platform);

  return 0;
}
#endif
