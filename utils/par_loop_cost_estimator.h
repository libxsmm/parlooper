/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef _PAR_LOOP_COST_ESIMATOR_H_
#define _PAR_LOOP_COST_ESIMATOR_H_
#include <string>
#include <vector>

typedef enum cost_analysis_type {
  SINGLE_TRACE = 1,
  PARALLEL_TRACES = 2,
  CONCURRENT_TRACES = 3
} cost_analysis_type;

typedef enum mem_hierarchy_type {
  MEM_L2_L1 = 1,
  MEM_LLC_L2_L1 = 2
} mem_hierarchy_type;

typedef enum platform_type { CLX = 1 } platform_type;

typedef enum mem_hierarchy_loc {
  L1 = 1,
  L2 = 2,
  LLC = 3,
  MEM = 4
} mem_hierarchy_loc;

typedef struct data_volume_t {
  int l1_bytes;
  int l2_bytes;
  int llc_bytes;
  int mem_bytes;
  int mixed_mem_llc_bytes;
} data_volume_t;

typedef struct platform_spec_t {
  double mixed_mem_llc_bytes_per_cycle;
  double mem_bytes_per_cycle;
  double llc_bytes_per_cycle;
  double l2_bytes_per_cycle;
  double l1_bytes_per_cycle;
  double llc_size_in_bytes;
  double l2_size_in_bytes;
  double l1_size_in_bytes;
  int n_threads;
  mem_hierarchy_type mem_hierarchy;
  double flops_per_cycle_out_of_L1;
  double bf16_flops_per_cycle_out_of_L2;
  double fp32_flops_per_cycle_out_of_L2;
  double freq_in_ghz;
} platform_spec_t;

typedef struct tensor_metadata_t {
  int subtensor_a_size_bytes;
  int subtensor_b_size_bytes;
  int subtensor_c_size_bytes;
  int M;
  int N;
  int K;
  int brcount;
  int dtype_size;
} tensor_metadata_t;

void set_platform_specs(
    platform_type platform,
    int n_threads,
    platform_spec_t* platform_specs);

void set_tensor_metadata(
    int M,
    int N,
    int K,
    int brcount,
    int dtype_size,
    tensor_metadata_t* tensor_metadata);

double tensor_contraction_cost_estimator(
    cost_analysis_type analysis_type,
    std::vector<std::string>* traces_array,
    tensor_metadata_t tensor_metadata,
    platform_spec_t platform_spec);

#endif // _PAR_LOOP_COST_ESIMATOR_H_
