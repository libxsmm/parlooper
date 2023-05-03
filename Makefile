###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
.SUFFIXES:

MAKEFLAGS += --no-print-directory
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

BLDDIR := $(if $(BLDDIR),$(BLDDIR),./build)
LIBDIR := $(if $(LIBDIR),$(LIBDIR),./lib)
LIBXSMM_ROOT := $(if $(LIBXSMM_ROOT),$(LIBXSMM_ROOT),../../libxsmm)
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
IFLAGS = -I./include -I./utils -I$(LIBXSMM_ROOT)/include
SRCDIRS = ./utils ./src .
SRCFILES := jit_compile.cpp par_loop_generator.cpp
OBJFILES := $(patsubst %,$(BLDDIR)/%,$(notdir $(SRCFILES:.cpp=-cpp.o)))
vpath %.cpp $(SRCDIRS) 

#$(info "BLDDIR = $(BLDDIR)")
#$(info "OBJFILES = $(OBJFILES)")

.PHONY: all
# To avoid dealing with dependency files or having stale object files, object files are always cleaned up after each build
all: builddir libdir slib dlib
	rm -f $(BLDDIR)/*.o

builddir:
	mkdir -p $(BLDDIR)

libdir:
	mkdir -p $(LIBDIR)

slib: libdir $(OBJFILES)
	ar rcs $(LIBDIR)/libparlooper.a $(OBJFILES)

dlib: libdir $(OBJFILES)
	$(CXX) -o $(LIBDIR)/libparlooper.so $(OBJFILES) -shared

$(BLDDIR)/%-cpp.o: %.cpp
	$(CXX) -fPIC $(CXXFLAGS) $(IFLAGS) $(LFLAGS)  -c $< -o $@ $(LDFLAGS)

CLEANOBJ:
	rm -f $(BLDDIR)/*.o

.PHONY: clean
clean: CLEANOBJ
	rm -f $(LIBDIR)/*.a $(LIBDIR)/*.so

