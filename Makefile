###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
CXXFLAGS = -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 
LDFLAGS = -ldl -lxsmm -lxsmmnoblas
IFLAGS = -I./libxsmm/include -I./libxsmm/samples/deeplearning/libxsmm_dnn/include/
LFLAGS = -L./libxsmm/lib/
SRCFILES = common_loops.cpp par_loop_cost_estimator.cpp par_loop_generator.cpp jit_compile.cpp

XFILES := $(OUTDIR)/conv_bwd $(OUTDIR)/conv_upd $(OUTDIR)/gemm $(OUTDIR)/gemm_bwd $(OUTDIR)/gemm_upd $(OUTDIR)/loop_permute_generator $(OUTDIR)/conv_fwd

.PHONY: all
all: $(XFILES)

$(OUTDIR)/gemm_upd:
	g++  gemm_model_upd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o gemm_upd

$(OUTDIR)/gemm_bwd:
	g++  gemm_model_bwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o gemm_bwd

$(OUTDIR)/gemm:
	g++  gemm_model_fwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o gemm

$(OUTDIR)/conv_fwd:
	g++ conv_model_fwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_fwd

$(OUTDIR)/conv_bwd:
	g++ conv_model_bwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_bwd

$(OUTDIR)/conv_upd:
	g++ conv_model_upd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_upd

$(OUTDIR)/loop_permute_generator:
	g++ $(CXXFLAGS) spec_loop_generator.cpp -o loop_permute_generator

.PHONY: clean
clean:
	rm -rf gemm gemm_bwd gemm_upd loop_permute_generator conv_fwd conv_bwd conv_upd
