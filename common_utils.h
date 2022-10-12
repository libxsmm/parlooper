/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <stdio.h>
#include <array>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <functional>
#include <omp.h>
#include "jit_compile.h"
#include "par_loop_cost_estimator.h"
#include "par_loop_generator.h"
#include "threaded_loops.h"
#include <cstring>
#include <unistd.h>
#include <libxsmm.h>
#include <dnn_common.h>
#define AARCH64_RDTSC

double ifreq;

#ifdef AARCH64_RDTSC
static __inline__ unsigned long long rdtsc(void) {
  unsigned long long virtual_timer_value;
  asm volatile("mrs %0, cntvct_el0" : "=r"(virtual_timer_value));
  return virtual_timer_value;
}
#else
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#endif

inline double getFreq() {
  long long int s = rdtsc();
  sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

void find_prime_factors(long num, std::vector<long>& res) {
  long n = num;
  if (n == 1) {
    res.push_back(n);   
  }
  while (n % 2 == 0) {
    res.push_back(2);
    n = n/2;
  }
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    while (n % i == 0) {
      res.push_back(i);
      n = n/i;
    }
  }
  if (n > 2) {
    res.push_back(n);
  }
  return;
}
