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
LDFLAGS = -ldl 
IFLAGS = -I$(PARLOOPER_ROOT)/include -I$(PARLOOPER_ROOT)/utils -I$(LIBXSMM_ROOT)/include -I$(LIBXSMM_DNN_ROOT)/include
SRCDIRS = $(PARLOOPER_ROOT)/utils $(PARLOOPER_ROOT)/src .
#SRCFILES := par_loop_generator.cpp jit_compile.cpp common_loops.cpp spgemm.cpp
#OBJFILES := $(patsubst %,$(BLDDIR)/%,$(notdir $(SRCFILES:.cpp=-cpp.o)))
XFILES := $(OUTDIR)/spgemm 
vpath %.cpp $(SRCDIRS) 

.PHONY: all
all: $(XFILES) CLEANOBJ

$(BLDDIR)/%-cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -c $< -o $@

$(OUTDIR)/spgemm: $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/spgemm-cpp.o
	$(CXX) $(BLDDIR)/common_loops-cpp.o  $(BLDDIR)/spgemm-cpp.o $(PARLOOPER_ROOT)/lib/libparlooper.a $(LIBXSMM_ROOT)/lib/libxsmm.a $(LIBXSMM_ROOT)/lib/libxsmmnoblas.a $(CXXFLAGS) $(IFLAGS) $(LFLAGS) $(LDFLAGS) -o spgemm

CLEANOBJ:
	rm -f *.o $(BLD_DIR)/*.o

.PHONY: clean
clean: CLEANOBJ
	rm -f $(XFILES)
