###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
LIBXSMM_ROOT := $(if $(LIBXSMM_ROOT),$(LIBXSMM_ROOT),./libxsmm/)
LIBXSMM_DNN_ROOT := $(if $(LIBXSMM_DNN_ROOT),$(LIBXSMM_DNN_ROOT),./libxsmm_dnn/)

CXX=icpc
CXXFLAGS = -Wno-format -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 
LDFLAGS = -ldl -lxsmm -lxsmmnoblas
IFLAGS = -I$(LIBXSMM_ROOT)/include -I$(LIBXSMM_DNN_ROOT)/include -I$(LIBXSMM_ROOT)/samples/deeplearning/libxsmm_dnn/include/
LFLAGS = -L$(LIBXSMM_ROOT)/lib/
SRCFILES = common_loops.cpp par_loop_cost_estimator.cpp par_loop_generator.cpp jit_compile.cpp

XFILES := $(OUTDIR)/conv_bwd $(OUTDIR)/conv_upd $(OUTDIR)/gemm $(OUTDIR)/gemm_bwd $(OUTDIR)/gemm_upd $(OUTDIR)/loop_permute_generator $(OUTDIR)/conv_fwd

.PHONY: all
all: $(XFILES)

$(OUTDIR)/gemm_upd:
	${CXX}  gemm_model_upd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o gemm_upd

$(OUTDIR)/gemm_bwd:
	${CXX}  gemm_model_bwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o gemm_bwd

$(OUTDIR)/gemm:
	${CXX}  gemm_model_fwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o gemm

$(OUTDIR)/conv_fwd:
	${CXX} conv_model_fwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_fwd

$(OUTDIR)/conv_bwd:
	${CXX} conv_model_bwd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_bwd

$(OUTDIR)/conv_upd:
	${CXX} conv_model_upd.cpp $(SRCFILES) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o conv_upd

$(OUTDIR)/loop_permute_generator:
	${CXX} $(CXXFLAGS) spec_loop_generator.cpp -o loop_permute_generator

.PHONY: clean
clean:
	rm -rf gemm gemm_bwd gemm_upd loop_permute_generator conv_fwd conv_bwd conv_upd
