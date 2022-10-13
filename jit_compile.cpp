/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "jit_compile.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

void* jit_compile_and_load(
    const std::string filename,
    const std::string flags) {
  char libname[] = "/tmp/ppx_XXXXXX";
  int fd = mkstemp(libname);
  unlink(libname);
  char fdname[50];
  sprintf(fdname, "/proc/self/fd/%d", fd);
  auto cmd = std::string("icpc -shared -fPIC -x c++ ") + flags;
  cmd = cmd + " -o " + libname + " " + filename;
  printf("JIT COMPILE: %s\n", cmd.c_str());
  int ret = system(cmd.c_str());
  if (ret != 0)
    return NULL;
  auto handle = dlopen(libname, RTLD_LAZY | RTLD_NODELETE);
  if (!handle) {
    fputs(dlerror(), stderr);
    return NULL;
  }
  return handle;
}

void* jit_from_file(
    const std::string filename,
    const std::string flags,
    const std::string func_name) {
  void* handle = jit_compile_and_load(filename, flags);
  if (handle == NULL)
    return NULL;
  void* func = dlsym(handle, func_name.c_str());
  if (func == NULL) {
    printf("Unable to find '%s' symbol in JIT COMPILE\n", func_name.c_str());
  }
  dlclose(handle);
  return func;
}

void* jit_from_str(
    const std::string src,
    const std::string flags,
    const std::string func_name) {
  char filename[] = "/tmp/ppx_XXXXXX";
  int fd = mkstemp(filename);
  unlink(filename);
  char fdname[50];
  sprintf(fdname, "/proc/self/fd/%d", fd);
  write(fd, src.c_str(), src.length());
  return jit_from_file(fdname, flags, func_name);
}
