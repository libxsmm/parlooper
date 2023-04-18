import subprocess
import os
import sys

import itertools

from contextlib import redirect_stdout

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

#import torch
import numpy as np
import time

script_version='v1'

# A helper
def for_recursive(number_of_loops, range_list, execute_function, current_index=0, iter_list = [], **kwargs):
    #print("range_list, current_index = ", range_list, current_index)
    if iter_list == []:
        iter_list = [0]*number_of_loops

    if current_index == number_of_loops-1:
        for iter_list[current_index] in range_list[current_index]:
            execute_function(iter_list, **kwargs)
    else:
        for iter_list[current_index] in range_list[current_index]:
            for_recursive(number_of_loops, iter_list = iter_list, range_list = range_list,  current_index = current_index+1, execute_function = execute_function, **kwargs)

# Usage
#def do_whatever(index_list):
#    return print(index_list)
#for_recursive(range_list = [range(0,3), range(0,3), range(1,3)], execute_function = do_whatever , number_of_loops=3)
#for_recursive(range_list = [range(0,3), range(0,3), range(1,3)], execute_function = do_whatever , number_of_loops=3)
#for_recursive(range_list = [range(0,1), range(0,2), range(0,1)], execute_function = do_whatever, number_of_loops=3)
#exit()

dbg_counter = 0

def xbf_tester(index_list, loop_string=None,
                cbfs=None, kbfs=None, hbfs=None, wbfs=None,
                basic_params=None, bc=None, bk=None,
                hs_in_gemm=None, pack_input=None,
                niters=20, redirect_output=None): # **kwargs):
    #print("dbg: index_list = ", index_list)
    if pack_input is None:
        pack_input = 0

    if cbfs is not None:
        #c_blocks = [ cbfs[i][index_list[j]] for (i,j) in zip([0, 1, 2, 3], [0, 1, 2, 3])]
        c_block = cbfs[index_list[0]]
    else:
        c_block = 1

    if kbfs is not None:
        k_block = kbfs[index_list[1]]
    else:
        k_block = 1

    if hbfs is not None:
        h_block = hbfs[index_list[2]]
    else:
        h_block = 1

    if wbfs is not None:
        w_block = wbfs[index_list[3]]
    else:
        w_block = 1

    if hs_in_gemm is not None:
        h_in_gemm = hs_in_gemm[index_list[4]]
    else:
        h_in_gemm = 1

    tuning_params = [ h_block, w_block, c_block, k_block, h_in_gemm, pack_input ]

    tuning_string = [ loop_string ]

    print("dbg: tuning_params = ", tuning_params)
    print("dbg: tuning_string = ", tuning_string)

    HBFcount = loop_string.count('d')

    if HBFcount > 1 and (h_block % h_in_gemm != 0):
        print("bad config, h_in_gemm does not divide h_block as loop string, h blocking factor count, h_block, h_in_gemm = ", tuning_string, HBFcount, h_block, h_in_gemm)
    else:
        """
        #run_test_bottleneck(*nhwck_params, bs, bs, bs, bs, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm,
        #                     opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module,
        #                     tuning_params, tuning_strings, niters)
        """

        [batch, H, W, C, K, R, S, stride, padding] = basic_params
        #OMP_NUM_THREADS=$omp USE_BF16=1 ./run_conv_upd.sh Aefcdb $batch 7 7 512 2048  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
        #OMP_NUM_THREADS=56 USE_BF16=1 ${preamble} ./conv_fwd A{C:14}C{R:4}fgbde  56 7 7 512 2048 1 1  1 1 0 0  32 32  1 1 1 1 1 0 1000 0 0
        basic_params_as_string = str(batch) + ' ' + str(H) + ' ' + str(W) + ' ' + str(C) + ' ' + str(K) + ' ' + str(R) + ' ' + str(S) + ' ' + str(stride) + ' ' + str(stride) + ' ' + str(padding) + ' ' + str(padding)
        tuning_params_as_string = str(h_block) + ' ' + str(w_block) + ' ' + str(c_block) + ' ' + str(k_block) + ' ' + str(h_in_gemm) + ' ' + str(pack_input)
        #print("dbg: bc, bk = ", bc, bk)
        cmd = '( ./conv_fwd ' + loop_string + ' ' + basic_params_as_string + ' '  + str(bc) + ' ' + str(bk) + ' ' + tuning_params_as_string + ' ' + str(niters) + ' )'
        if redirect_output is not None:
            cmd = cmd + ' >> ' + redirect_output + ' '
        print("cmd to execute = ", cmd)
        #exit()
        os.system(cmd) # returns the exit status

        """
        global dbg_counter
        dbg_counter = dbg_counter + 1
        print("dbg: dbg_counter = ", dbg_counter)
        if dbg_counter == 3:
            print("dbg: exiting due to dbg_counter = ", dbg_counter)
            exit()
        """
        #exit()
        #return print(index_list)

#for_recursive(range_list = [range(0,1), range(0,2) , range(0,1)], execute_function = xbf_tester, number_of_loops=3)

loop_names = []
loop_specs = []

for i in range(1,2):
    for j in range(1,2):
        #loop_names.append('conv_upd_nchw_1x1_nohybrid')
        #loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K')
        #loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + 'd_0_K,' + 'e_0_K,' + 'f_0_K')
        #loop_names.append('conv_upd_nchw_3x3_nohybrid') # will use different skipping condition potentially than 1x1
        #loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K')
        #loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + 'd_0_K,' + 'e_0_K,' + 'f_0_K')
        #loop_names.append('conv_upd_chwn')
        #loop_specs.append('A_0_M,' + 'B_0_M,' + 'c_0_K,' + 'd_0_K,' + 'e_0_M,' + 'f_0_M')
        loop_names.append('conv_fwd_general')
        loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_K,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')
        loop_names.append('conv_fwd_hybrid')
        loop_specs.append('A_0_M,' + 'b_0_K,' + 'C_0_M,' + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')
        #loop_specs.append('A_0_M,' + 'b_0_K,' + ('C_' + str(i) + '_M,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K') # showed poor performance for 7x7 as it ran out of parallelism
        #loop_names.append('bottleneck_last')
        #loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_M,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')

print("dbg: loop_names: ", loop_names)
print("dbg: loop_specs: ", loop_specs)

for i in range(len(loop_names)):
    cmd = './loop_permute_generator ' + loop_names[i] + ' ' + loop_specs[i] # extra optional arg for output file
    print("i, cmd to execute = ", i, cmd)
    os.system(cmd) # returns the exit status

#exit()

loop_lines = {}
nLoops = {}

for name in loop_names:
    print("dbg: loop_name = ", name)
    cmd = 'cat ' + name + '_bench_configs.txt > uber_config.txt'
    print("dbg: ", cmd)
    os.system(cmd) # returns the exit status
    cmd = "awk '!seen[$0]++' uber_config.txt > tuner_config.txt"
    print("dbg: ", cmd)
    os.system(cmd) # returns the exit status
    cmd = 'rm uber_config.txt'
    print("dbg: ", cmd)
    os.system(cmd) # returns the exit status

#cat *bench_configs.txt > uber_config.txt
#awk '!seen[$0]++' uber_config.txt > tuner_config.txt
#rm uber_config.txt

#exit()

    with open("tuner_config.txt",'r') as f:
        loop_lines[name] = f.read().splitlines()
        nLoops[name] = len(loop_lines[name])
        print("dbg: total number of loop lines for name = ", name, nLoops[name])

#exit()

nthreads=int(os.getenv("OMP_NUM_THREADS"))
print("dbg: nthreads = ", nthreads)

#exit()

#for x in list(range(0,2)) + [4, 5]:
#  print("x =", x)
#exit()

# l = 10 * (bottleneck_number + 1) + convolution index within the bottleneck (1-3 or 1-4 when residual connection is present)
#for l in [81]: #[81, 82, 83]:
for l in list(range(1, 20)) + [20, 21, 22] + [206, 211, 216]:
#for l in [17, 18, 22, 5]:
#for l in [18, 22, 5]:
#for l in [10]:
    file_path = 'conv_fwd_' + str(l) + '_tuning_dbg.txt'
    sys.stdout = open(file_path, "w")
    #file_path = None

    if True: #with open(file_path, 'w') as f:
        if True: #with redirect_stdout(f):

            # common parameters (potentially overwrite-able)
            N=nthreads
            bc=32
            bk=32
            #bc=32
            #bk=32
            niters=1000

            n_img_teams=1
            n_ofm_teams=1

            print("l = ", l)

            # 19 convolutions used for ADL measurements
            if l == 0:
                basic_params = [N, 224, 224, 3, 64, 7, 7, 2, 3]
                KBFS=None #[[1]]
                CBFS=None #[[1]]
                HBFS=None #[4, 7, 14, 28]
                WBFS=[1, 2, 4, 7]
                hs_in_gemm=None #[1, 2, 4, 7]
                config_name='conv_fwd_general'
                bc=3
            elif l == 1:
                basic_params = [N, 56, 56, 64, 256, 1, 1, 1, 0]
                KBFS=None #[[1]]
                CBFS=None #[[1]]
                HBFS=[4, 7, 14]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 2:
                basic_params = [N, 56, 56, 64, 64, 1, 1, 1, 0]
                KBFS=None #[[1]]
                CBFS=None #[[1]]
                HBFS=[2, 4, 7, 14]
                WBFS=[1, 2, 4, 8]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 3:
                basic_params = [N, 56, 56, 64, 64, 3, 3, 1, 1]
                KBFS=None
                CBFS=None
                HBFS=[2, 4, 7, 14]
                WBFS=[1, 2, 4, 8]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 4:
                basic_params = [N, 56, 56, 256, 64, 1, 1, 1, 0]
                KBFS=None
                CBFS=[1, 2, 8]
                HBFS=[2, 4]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 5:
                basic_params = [N, 56, 56, 256, 512, 1, 1, 2, 0]
                KBFS=None
                CBFS=[1, 2, 4]
                HBFS=[2, 4]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 6:
                basic_params = [N, 56, 56, 256, 128, 1, 1, 2, 0]
                KBFS=None
                CBFS=[1, 2, 4]
                HBFS=[2, 4]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 7:
                basic_params = [N, 28, 28, 128, 128, 3, 3, 1, 1]
                KBFS=None
                CBFS=None
                HBFS=[2, 4, 7, 14]
                WBFS=[1, 2, 4]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 8:
                basic_params = [N, 28, 28, 128, 512, 1, 1, 1, 0]
                KBFS=None
                CBFS=[1, 2, 4]
                HBFS=[4, 7]
                WBFS=[1, 2]
                hs_in_gemm=None #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 9:
                basic_params = [N, 28, 28, 512, 128, 1, 1, 1, 0]
                KBFS=None
                CBFS=[1, 4, 8, 16]
                HBFS=[4, 7]
                WBFS=[1, 2]
                hs_in_gemm=None #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 10:
                basic_params = [N, 28, 28, 512, 1024, 1, 1, 2, 0]
                KBFS=[8]
                CBFS=[1, 4, 8, 16]
                HBFS=[4, 7]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2] #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 11:
                basic_params = [N, 28, 28, 512, 256, 1, 1, 2, 0]
                KBFS=[2, 4]
                CBFS=[1, 4, 8, 16]
                HBFS=[4, 7]
                WBFS=[1, 2]
                hs_in_gemm=[1 ,2] #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 12:
                basic_params = [N, 14, 14, 256, 256, 3, 3, 1, 1]
                KBFS=None
                CBFS=[1, 2, 4]
                HBFS=[2, 7]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4] #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 13:
                basic_params = [N, 14, 14, 256, 1024, 1, 1, 1, 0]
                KBFS=[2, 4, 8]
                CBFS=[1, 2, 4]
                HBFS=[2, 7]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2] #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 14:
                basic_params = [N, 14, 14, 1024, 256, 1, 1, 1, 0]
                KBFS=[2, 4, 8]
                CBFS=[1, 4, 8, 16]
                HBFS=[2, 7]
                WBFS=[1, 2]
                hs_in_gemm=[1, 2] #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 15:
                basic_params = [N, 14, 14, 1024, 2048, 1, 1, 2, 0]
                KBFS=[2, 4, 8]
                CBFS=[1, 2, 4, 8] # reduce to 1, 2, 4 for shorter time
                HBFS=[1, 2, 7]    # reduce to 1, 2 for shorter time
                WBFS=None
                hs_in_gemm=[1, 7] #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 16:
                basic_params = [N, 14, 14, 1024, 512, 1, 1, 2, 0]
                KBFS=[2, 4, 8]
                CBFS=[1, 2, 4, 8] # reduce to 1, 2, 4 for shorter time
                HBFS=[1, 2, 7]    # reduce to 1, 2 for shorter time
                WBFS=None
                hs_in_gemm=[1, 7] #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 17:
                basic_params = [N, 7, 7, 512, 512, 3, 3, 1, 1]
                KBFS=[2, 4, 8]
                CBFS=[1, 4, 8, 16]
                HBFS=None
                WBFS=None
                hs_in_gemm=[1, 7]
                config_name='conv_fwd_hybrid'

            elif l == 18:
                basic_params = [N, 7, 7, 512, 2048, 1, 1, 1, 0]
                KBFS=[2, 4, 8, 16]
                CBFS=[1, 4, 8]
                HBFS=None
                WBFS=None
                hs_in_gemm=[1, 7]
                config_name='conv_fwd_hybrid'

            elif l == 19:
                basic_params = [N, 7, 7, 2048, 512, 1, 1, 1, 0]
                KBFS=[2, 4, 8]
                CBFS=[1, 4, 8, 16]
                HBFS=None
                WBFS=None
                hs_in_gemm=[1, 7]
                config_name='conv_fwd_hybrid'

            # Resnet-50 sizes for e2e training (additional to ADL sizes)

            elif l == 20:
                basic_params = [N, 56, 56, 128, 128, 3, 3, 2, 1]
                KBFS=None
                CBFS=None
                HBFS=[4, 7]
                WBFS=None
                hs_in_gemm=None
                config_name='conv_fwd_general'

            elif l == 21:
                basic_params = [N, 28, 28, 256, 256, 3, 3, 2, 1]
                KBFS=[2, 4, 8]
                CBFS=None #[1, 4, 8, 16]
                HBFS=None
                WBFS=None
                hs_in_gemm=None
                config_name='conv_fwd_general'

            elif l == 22:
                basic_params = [N, 14, 14, 512, 512, 3, 3, 2, 1]
                KBFS=[2, 4, 8] #[2, 4, 8, 16]
                CBFS=None #[1, 4, 8, 16]
                HBFS=None
                WBFS=None
                hs_in_gemm=None
                config_name='conv_fwd_general'

            elif l == 206: # Same as l = 6 but with stride = 1. Not sure why in ADL sizes it has stride 2
                basic_params = [N, 56, 56, 256, 128, 1, 1, 1, 0]
                KBFS=None
                CBFS=None
                HBFS=[4, 7]
                WBFS=[1, 2]
                hs_in_gemm=None
                config_name='conv_fwd_general'

            elif l == 211: # Same as l = 11 but with stride = 1. Not sure why in ADL sizes it has stride 2
                basic_params = [N, 28, 28, 512, 256, 1, 1, 1, 0]
                KBFS=None
                CBFS=[1, 4, 8]
                HBFS=None
                WBFS=None
                hs_in_gemm=None #[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 216: # Same as l = 16 b ut with stride = 1. Not sure why in ADL sizes it has stride 2
                basic_params = [N, 14, 14, 1024, 512, 1, 1, 1, 0]
                KBFS=[2, 4, 8, 16]
                CBFS=None
                HBFS=None
                WBFS=None
                hs_in_gemm=[1, 2]
                config_name='conv_fwd_general'

            # Resnet-50 sizes for e2e training (in the order of topology fwd) [UNFINISHED]

            elif l == 111: #! covered by the ADL list (l = 2)
                basic_params = [N, 56, 56, 64, 64, 1, 1, 1, 0]
                KBFS=None #[[1]]
                CBFS=None #[[1]]
                HBFS=[4, 7] # 14
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'
            elif l == 112: #! covered by the ADL list (l = 3)
                basic_params = [N, 56, 56, 64, 64, 3, 3, 1, 1]
                KBFS=None #[[1]]
                HBFS=[4, 7] # 14
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'
            elif l == 113: #! covered by the ADL list (l = 4)
                basic_params = [N, 56, 56, 64, 256, 1, 1, 1, 0]
                KBFS=None #[[1]]
                CBFS=None #[[1]]
                HBFS=[4, 7] # 14
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'
            elif l == 114: # a repetition of l = 13 = covered by the ADL list
                basic_params = [N, 56, 56, 64, 256, 1, 1, 1, 0]
                KBFS=None #[[1]]
                CBFS=None #[[1]]
                HBFS=[4, 7] # 14
                WBFS=[1, 2]
                hs_in_gemm=[1, 2, 4, 7]
                config_name='conv_fwd_general'

            elif l == 121: #! repitition of l = 11 = covered by the ADL list (l = 2)
                basic_params = [N, 56, 56, 256, 64, 1, 1, 1, 0]
                config_name='conv_fwd_general'
            elif l == 122: #! repitition of l = 112 = covered by the ADL list (l = 3)
                basic_params = [N, 56, 56, 64, 64, 3, 3, 1, 1]
                config_name='conv_fwd_general'
            elif l == 123: #! repitition of l = 113 = covered by the ADL list (l = 4)
                basic_params = [N, 56, 56, 64, 256, 1, 1, 1, 0]
                config_name='conv_fwd_general'
                """
            elif l == 131: #! repitition of l = 11 = covered by the ADL list (l = 2)
                basic_params = [N, 56, 56, 256, 128, 1, 1, 1, 0]
                config_name='conv_fwd_general'
            elif l == 132: #! repitition of l = 112 = covered by the ADL list (l = 3)
                basic_params = [N, 56, 56, 64, 64, 3, 3, 1, 1]
                config_name='conv_fwd_general'
            elif l == 133: #! repitition of l = 113 = covered by the ADL list (l = 4)
                basic_params = [N, 56, 56, 64, 256, 1, 1, 1, 0]
                config_name='conv_fwd_general'
            elif l == 134: #! repitition of l = 113 = covered by the ADL list (l = 4)
                basic_params = [N, 56, 56, 64, 256, 1, 1, 1, 0]
                config_name='conv_fwd_general'
                """

            elif l == 181: #! covered by the ADL list (l = 18)
                basic_params = [N, 7, 7, 512, 2048, 1, 1, 1, 0]
                KBFS=[2, 4, 8]
                CBFS=None #[[1]]
                HBFS=None
                WBFS=None
                hs_in_gemm=[1, 7]
                config_name='conv_fwd_hybrid'
            elif l == 182: #! covered by the ADL list (l = 17)
                basic_params = [N, 7, 7, 512, 512, 3, 3, 1, 1]
                KBFS=[2, 4, 8]
                CBFS=None #[[1]]
                HBFS=None
                WBFS=None
                hs_in_gemm=[1, 7]
                config_name='conv_fwd_hybrid'
            elif l == 183: #! covered by the ADL list (l = 19)
                basic_params = [N, 7, 7, 2048, 512, 1, 1, 1, 0]
                KBFS=[2, 4, 8, 16]
                CBFS=None #[[1]]
                HBFS=None
                WBFS=None
                hs_in_gemm=[1, 7]
                config_name='conv_fwd_hybrid'
            else:
                  print("Error: index l does not match any existing cases, l = ", l)
                  exit()

            ncombinations = 0
            nlinecombs = 0
            nlines = 0

            # For strided 1x1 convolutions, specifically, pack_input must be allowed to be 1
            [batch, H, W, C, K, R, S, stride, padding] = basic_params
            if stride == 2:
                range_for_pack = range(1, 2)
            else:
                range_for_pack = range(0, 1) #pack_input_upfront_limit + 1)

            for line in loop_lines[config_name]:
                if config_name == 'conv_fwd_general' and not line.startswith('Afgb'):
                    continue
                elif config_name == 'conv_fwd_hybrid' and ((not line.startswith('AC') and not line.startswith('CA')) or 'fgb' not in line):
                    continue
                else:
                    print("line = ", line)
                    CBFcount = line.count('b') #$( echo ${lowerline} | tr -d -c 'c' | awk '{ print length; }' )
                    print("CBFcount = ", CBFcount)
                    KBFcount = line.count('c') #$( echo ${lowerline} | tr -d -c 'c' | awk '{ print length; }' )
                    print("KBFcount = ", KBFcount)
                    #echo "C count is ${KBFcount}"
                    HBFcount = line.count('d') #$( echo ${lowerline} | tr -d -c 'd' | awk '{ print length; }' )
                    print("HBFcount = ", HBFcount)
                    WBFcount = line.count('e') #$( echo ${lowerline} | tr -d -c 'd' | awk '{ print length; }' )
                    print("WBFcount = ", WBFcount)
                    nlines = nlines + 1

                    if CBFcount > 1 and CBFS == None:
                        continue
                    elif KBFcount > 1 and KBFS == None:
                        continue
                    elif HBFcount > 1 and HBFS == None:
                        continue
                    elif WBFcount > 1 and WBFS == None:
                        continue

                # teams_pairs = a set of pairs [n_img_teams, n_ofm_teams]
                if config_name == 'conv_fwd_hybrid':
                    if nthreads == 28:
                        teams_pairs = [[14, 2], [7, 4]]
                    elif nthreads == 56:
                        teams_pairs = [[56, 1], [14, 4], [7, 8], [1, 56]]
                    elif nthreads == 52:
                        teams_pairs = [[13, 4]]
                    elif nthreads == 16:
                        teams_pairs = [[4, 4], [16, 1], [1, 16]]
                    elif nthreads == 64:
                        teams_pairs = [[64, 1], [32, 2], [16, 4], [8, 8], [4, 16], [2, 32], [1, 64]]
                else:
                    teams_pairs = [[1, 1]]

                for teams_pair in teams_pairs:
                    [n_img_teams, n_ofm_teams] = teams_pair

                    # Necessary pre-processing for explicit 2d parallelization
                    if config_name == 'conv_fwd_hybrid':
                        # Caution: order of replace() calls matter here!
                        line_with_teams = line
                        line_with_teams = line_with_teams.replace('C', 'C{R:' + str(n_ofm_teams) + '}', 1)
                        line_with_teams = line_with_teams.replace('A', 'A{C:' + str(n_img_teams) + '}', 1)
                        print("dbg:modified with teams line = ", line_with_teams)
                    else:
                        line_with_teams = line

                    #for_recursive(range_list = [range(0,3), range(0,3) , range(1,3)], execute_function = do_whatever , number_of_loops=3)
                    #print('dbg: basic_params[7] = ', basic_params[7])
                    #print('dbg: basic_params[8] = ', basic_params[8])
                    if basic_params[7] == 2:
                        pack_input = 1
                    else:
                        pack_input = 0

                    for pack_input_var in range(0, pack_input + 1):

                        if stride == 2 and pack_input_var == 0 and hs_in_gemm != None:
                            use_hs_in_gemm = None
                            #continue
                        else:
                            use_hs_in_gemm = hs_in_gemm
                        nbfloops = 0

                        if KBFcount == 1 or KBFS is None:
                            use_kbfs = None
                        else:
                            use_kbfs = KBFS
                        nbfloops = nbfloops + 1

                        if HBFcount == 1 or HBFS is None:
                            use_hbfs = None
                        else:
                            use_hbfs = HBFS
                        nbfloops = nbfloops + 1

                        # CBFS and WBFS are different from KBFS and HBFS as there are not explicitly used as blocking factors in the loop strings
                        use_cbfs = CBFS
                        nbfloops = nbfloops + 1

                        use_wbfs = WBFS
                        nbfloops = nbfloops + 1

                        cf_ranges = [range(0, len(use_cbfs))] if use_cbfs is not None else [range(0,1)]
                        kf_ranges = [range(0, len(use_kbfs))] if use_kbfs is not None else [range(0,1)]
                        hf_ranges = [range(0, len(use_hbfs))] if use_hbfs is not None else [range(0,1)]
                        wf_ranges = [range(0, len(use_wbfs))] if use_wbfs is not None else [range(0,1)]

                        h_in_gemm_range = range(0, len(use_hs_in_gemm)) if use_hs_in_gemm is not None else range(1,2)
                        nbfloops = nbfloops + 1

                        #print("hs_in_gemm = ", hs_in_gemm)
                        #print("use_hs_in_gemm = ", use_hs_in_gemm)

                        print("range_list = ", [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges, h_in_gemm_range])
                        #if pack_input_var == 1:
                        #    print("exiting due to the exit for pack_input_var = 1")
                        #    exit()

                        #print("dbg: bc, bk = ", bc, bk)

                        if HBFcount > 1:
                            n_good_pairs = len(*hf_ranges) * len(h_in_gemm_range)
                            #print("n_good_pairs before filtering = ", n_good_pairs)
                            for (i,j) in itertools.product(list(*hf_ranges), list(h_in_gemm_range)):
                                h_block = use_hbfs[i] if use_hbfs is not None else 1
                                h_in_gemm = use_hs_in_gemm[j] if use_hs_in_gemm is not None else 1
                                #print("(h_i, hingemm_j) = ", h_block, h_in_gemm)
                                if (h_block % h_in_gemm != 0):
                                    #print("bad pair, h_in_gemm does not divide h_block")
                                    n_good_pairs = n_good_pairs - 1
                            #print("n_good_pairs after filtering = ", n_good_pairs)
                            #exit()

                        for_recursive(range_list = [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges, h_in_gemm_range], execute_function = xbf_tester, number_of_loops = nbfloops,
                                      bk=bk, bc=bc,
                                      cbfs=use_cbfs, kbfs=use_kbfs, hbfs=use_hbfs, wbfs=use_wbfs,
                                      basic_params=basic_params, niters=niters,
                                      loop_string=line_with_teams,
                                      hs_in_gemm = use_hs_in_gemm,
                                      pack_input=pack_input_var,
                                      redirect_output=file_path,
                                      )

                        #ncombinations = ncombinations + sum([len[rangevar] for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
                        #print("range_list = ", [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges])
                        #print("its tmp list = ", [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
                        #print("its reduce product = ", reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]]))
                        #ncombinations = ncombinations + reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
                        if HBFcount > 1:
                            ncombinations = ncombinations + reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *wf_ranges]]) * n_good_pairs
                        else:
                            ncombinations = ncombinations + reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges, h_in_gemm_range]])
                        #exit()
                        print("")
            print("script version, l, config_name, nlines, ncombinations = ", script_version, l, config_name, nlines, ncombinations)
exit()
