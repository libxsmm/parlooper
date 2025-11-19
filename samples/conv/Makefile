###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
BLDDIR = .
OUTDIR = .
PARLOOPER_ROOT := $(if $(PARLOOPER_ROOT),$(PARLOOPER_ROOT),../../)
LIBXSMM_ROOT := $(if $(LIBXSMM_ROOT),$(LIBXSMM_ROOT),../../libxsmm)
LIBXSMM_DNN_ROOT := $(if $(LIBXSMM_DNN_ROOT),$(LIBXSMM_DNN_ROOT),../../libxsmm_dnn)
CXX = g++
CXXFLAGS = -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 
ifeq ($(PARLOOPER_COMPILER),gcc)
  CXX := g++
  CXXFLAGS := -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 
endif
ifeq ($(PARLOOPER_COMPILER),clang)
  CXX := clang++
  CXXFLAGS := -Wno-unused-command-line-argument -Wno-format -fopenmp=libomp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2  
endif
ifeq ($(PARLOOPER_COMPILER),icc)
  CXX := icpc
  CXXFLAGS := -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 
endif
ifeq ($(PARLOOPER_COMPILER),icx)
  CXX := icpx
  CXXFLAGS := -qopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 
endif
LDFLAGS = -ldl 
IFLAGS = -I$(PARLOOPER_ROOT)/include -I$(PARLOOPER_ROOT)/utils -I$(LIBXSMM_ROOT)/include -I$(LIBXSMM_DNN_ROOT)/include
SRCDIRS = $(PARLOOPER_ROOT)/utils $(PARLOOPER_ROOT)/src .
#SRCFILES := common_loops.cpp par_loop_cost_estimator.cpp
#OBJFILES := $(patsubst %,$(BLDDIR)/%,$(notdir $(SRCFILES:.cpp=-cpp.o)))
XFILES := $(OUTDIR)/conv_fwd $(OUTDIR)/conv_bwd $(OUTDIR)/conv_upd $(OUTDIR)/loop_permute_generator
vpath %.cpp $(SRCDIRS) 

.PHONY: all
all: $(XFILES) CLEANOBJ

$(BLDDIR)/%-cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -c $< -o $@

#$(OUTDIR)/conv_fwd: $(BLDDIR)/par_loop_cost_estimator-cpp.o  $(BLDDIR)/par_loop_generator-cpp.o $(BLDDIR)/jit_compile-cpp.o $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/conv_model_fwd-cpp.o
#	$(CXX) $(BLDDIR)/par_loop_cost_estimator-cpp.o  $(BLDDIR)/par_loop_generator-cpp.o $(BLDDIR)/jit_compile-cpp.o $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/conv_model_fwd-cpp.o $(LIBXSMM_ROOT)/lib/libxsmm.a $(LIBXSMM_ROOT)/lib/libxsmmnoblas.a $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_fwd
$(OUTDIR)/conv_fwd: $(BLDDIR)/par_loop_cost_estimator-cpp.o $(BLDDIR)/common_loops-cpp.o $(BLDDIR)/conv_model_fwd-cpp.o
	$(CXX) $(BLDDIR)/par_loop_cost_estimator-cpp.o $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/conv_model_fwd-cpp.o $(PARLOOPER_ROOT)/lib/libparlooper.a $(LIBXSMM_ROOT)/lib/libxsmm.a $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_fwd

$(OUTDIR)/conv_bwd: $(BLDDIR)/par_loop_cost_estimator-cpp.o $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/conv_model_bwd-cpp.o
	$(CXX) $(BLDDIR)/par_loop_cost_estimator-cpp.o $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/conv_model_bwd-cpp.o $(PARLOOPER_ROOT)/lib/libparlooper.a $(LIBXSMM_ROOT)/lib/libxsmm.a $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_bwd

$(OUTDIR)/conv_upd: $(BLDDIR)/par_loop_cost_estimator-cpp.o $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/conv_model_upd-cpp.o
	$(CXX) $(BLDDIR)/par_loop_cost_estimator-cpp.o $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/conv_model_upd-cpp.o $(PARLOOPER_ROOT)/lib/libparlooper.a $(LIBXSMM_ROOT)/lib/libxsmm.a $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_upd

$(OUTDIR)/loop_permute_generator:
	$(CXX) $(CXXFLAGS) $(PARLOOPER_ROOT)/utils/spec_loop_generator.cpp -o loop_permute_generator

CLEANOBJ:
	rm -f *.o $(BLDDIR)/*.o

.PHONY: clean
clean: CLEANOBJ
	rm -f $(XFILES)

