/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "common_utils.h"
#include "threaded_loops.h"
#include "gemm_common_utils.h"

#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <errno.h>
#include <string.h>

int main(int argc, char **argv) {
  // Setup default GEMM sizes
  long long M = 1024, N=1024, K=1024;
  long long bm = 32, bn = 32;
  double bw_per_core = 10.0; // in GB/s
  double c_bw_per_core = 10.0; // in GB/s
  double compute_per_core = 100.0; // in GFLOP/s
  double gflops_per_brgemm = 2.0 * ((double)bm * (double)bn * (double)K) / 1e9;
  double dtype_size_in_bytes = 2.0;
  double a_slice_size_in_gb = (double)bm * (double)K * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0); // in GB
  double b_slice_size_in_gb = (double)bn * (double)K * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0); // in GB
  double c_slice_size_in_gb = (double)bn * (double)bm * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0); // in GB
  long long threads = 64;
  long long m_teams = 1, n_teams = 1;
  long long c_copies_limit = 8;
  long long calculate_reduction_time = 0;

  // read inputs
  if (argc > 1) {
    threads = atoi(argv[1]);
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    bm = atoi(argv[5]);
    bn = atoi(argv[6]);
    bw_per_core = atof(argv[7]);
    c_bw_per_core = atof(argv[8]);
    compute_per_core = atof(argv[9]);
    c_copies_limit = atoi(argv[10]);
  }
  long long best_time_in_sec = 0;
  double best_roofline = 0.0;
  long long best_m_teams = 0;
  long long best_n_teams = 0;
  long long best_c_copies = 0;

  for (int ic = 1; ic <= c_copies_limit; ic = ic * 2) {
    if (threads % ic != 0) continue;
    long long threads_per_layer = threads / ic;
    long long m_tasks = M / bm;
    long long n_tasks = N / bn;
    long long K_per_layer = K / ic;
    gflops_per_brgemm = 2.0 * ((double)bm * (double)bn * (double)K_per_layer) / 1e9;
    calculate_reduction_time = (ic > 1) ? 1 : 0;
    for (int m_teams = 1; m_teams <= threads_per_layer; m_teams++) {
      if (threads_per_layer % m_teams != 0) continue;
      long long n_teams = threads_per_layer / m_teams;
      if (threads_per_layer % n_teams != 0) continue;
      long long m_tasks_per_team = (m_tasks + m_teams - 1) / m_teams;
      long long n_tasks_per_team = (n_tasks + n_teams - 1) / n_teams;
      long long brgemms_per_thread = m_tasks_per_team * n_tasks_per_team;
      long long brgemms_ab_mem = 1;
      long long brgemms_a_only_mem = m_tasks_per_team - 1;
      long long brgemms_b_only_mem = n_tasks_per_team - 1;
      long long brgemms_ab_l2 = brgemms_per_thread - brgemms_a_only_mem - brgemms_b_only_mem - brgemms_ab_mem;
      a_slice_size_in_gb = (double)bm * (double)K_per_layer * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0); // in GB
      b_slice_size_in_gb = (double)bn * (double)K_per_layer * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0); // in GB
      c_slice_size_in_gb = (double)bm * (double)bn * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0); // in GB
      double time_write_c_tile = 0;//c_slice_size_in_gb / c_bw_per_core;

      // Cycles for brgemms when a and b come from memory
      double time_brgemms_ab_mem = (brgemms_ab_mem * LIBXSMM_MAX(time_write_c_tile, (a_slice_size_in_gb + b_slice_size_in_gb) / bw_per_core));
      double time_brgemms_a_only_mem = (brgemms_a_only_mem * LIBXSMM_MAX(time_write_c_tile, (a_slice_size_in_gb) / bw_per_core));
      double time_brgemms_b_only_mem = (brgemms_b_only_mem * LIBXSMM_MAX(time_write_c_tile, (b_slice_size_in_gb) / bw_per_core));
      double time_brgemms_c_write = ((double)brgemms_per_thread * c_slice_size_in_gb / c_bw_per_core);
      double time_brgemms_compute = ((double)brgemms_ab_l2 * LIBXSMM_MAX(time_write_c_tile, gflops_per_brgemm / compute_per_core));
      double total_time_in_sec = time_brgemms_ab_mem + time_brgemms_a_only_mem + time_brgemms_b_only_mem + time_brgemms_c_write + time_brgemms_compute;

      if (calculate_reduction_time == 1) {
        // Add reduction time to c write time
        long long reduction_tasks = m_tasks * n_tasks;
        long long reduction_tasks_per_thread = (reduction_tasks + threads - 1) / threads;
        double reduction_volume_in_gb = (double)reduction_tasks_per_thread * (double)ic * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0); // in GB
        double reduction_time_in_sec = reduction_volume_in_gb / c_bw_per_core;
        total_time_in_sec += reduction_time_in_sec;
      }

      double total_gflops = (2.0 * (double)M * (double)N * (double)K) / 1e9;
      double roofline = total_gflops / total_time_in_sec;
      if (best_roofline < roofline) {
        best_roofline = roofline;
        best_time_in_sec = (long long)(total_time_in_sec * 1e9); // in ns
        best_m_teams = m_teams;
        best_n_teams = n_teams;
        best_c_copies = ic;
      }
    }
  }

  //printf("GEMM ROOFLINE MODEL RESULTS:\n");
  //printf("M=%lld, N=%lld, K=%lld, bm=%lld, bn=%lld, threads=%lld\n", M, N, K, bm, bn, threads);
  //printf("Best Roofline Performance: %.2f GFLOPS\n", best_roofline);
  //printf("Best Time Estimate: %lld ns\n", best_time_in_sec);
  //printf("Best m_teams: %lld, Best n_teams: %lld, C-copies: %lld\n", best_m_teams, best_n_teams, best_c_copies);
  // Single line output for easy grepping
  printf(" %lld x %lld x %lld ROOFLINE_MODEL %.2f GFLOPS m_teams=%lld n_teams=%lld c_copies=%lld\n", M, N, K, best_roofline, best_m_teams, best_n_teams, best_c_copies); 

  return 0;
}
