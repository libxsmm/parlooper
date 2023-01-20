/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef _JIT_COMPILE_H_
#define _JIT_COMPILE_H_

#include <string>

void* jit_from_file(
    const std::string filename,
    const std::string flags,
    const std::string func_name);

void* jit_from_str(
    const std::string src,
    const std::string flags,
    const std::string func_name);

#endif //_JIT_COMPILE_H_
