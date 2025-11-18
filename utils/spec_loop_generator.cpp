/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <functional>
#include <string.h>
#include <ctype.h>
#include <iostream>
using namespace std;

#define MAX_LOOP_DESC_SIZE 512
#define MAX_CODE_SIZE 1048576

typedef struct {
  long start;
  long end;
  long step;
  long block_size[8];
} loop_rt_spec_t;

typedef struct {
  char name;
  int  n_blocked_times;
  char type;
  char is_parallelizable;
  char is_currently_parallel_on;
  int  pos_first_occ;
} loop_metadata_t;

typedef struct {
  int  idx_id;
  char idx_name[256];
  char start_var_name[256];
  char end_var_name[256];
  char step_var_name[256];
  int  jit_start;
  int  jit_step;
  int  jit_end;
  int  jit_block_sizes;
  long start;
  long end;
  long step;
  int  pos_in_loopnest;
  int  is_parallelizable;
  int  is_blocked;
  int  is_blocked_outer;
  long block_size[8];
} loop_param_t;

typedef struct {
  char        *buf;
  int         cur_nest_level;
  int         cur_pos;
  int         n_loops;
  loop_param_t *loop_params;
  int         n_logical_loops;
  char        occurence_map[256];
  int         jit_loop_spec;
} loop_code;

loop_param_t find_loop_param_at_pos(loop_param_t *i_loop_params, int pos) {
  loop_param_t res;
  int i = 0;
  int found = 0;
  while (!found) {
    if (i_loop_params[i].pos_in_loopnest == pos) {
      found = 1;
      res = i_loop_params[i];
    } else {
      i++;
    }
  }
  return res;
}

void add_buf_to_code(loop_code *i_code, char *buf) {
  sprintf(i_code->buf + i_code->cur_pos, "%s", buf);
  i_code->cur_pos += strlen(buf);
}

void align_line(loop_code *i_code) {
  char tmp_buf[512];
  int i;
  for (i = 0; i < 2*i_code->cur_nest_level; i++) {
    tmp_buf[i] = ' ';
  }
  tmp_buf[2*i_code->cur_nest_level] = '\0';
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void increase_nest_level(loop_code *i_code) {
  i_code->cur_nest_level = i_code->cur_nest_level + 1;
  return;
}

void decrease_nest_level(loop_code *i_code) {
  i_code->cur_nest_level = i_code->cur_nest_level - 1;
  return;
}

void emit_parallel_for(loop_code *i_code, int collapse_level) {
  char tmp_buf[512];
  align_line(i_code);
  if (collapse_level > 1) {
    sprintf( tmp_buf, "#pragma omp for collapse(%d) nowait\n", collapse_level);
  } else {
    sprintf( tmp_buf, "#pragma omp for nowait\n");
  }
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_header(loop_code *i_code) {
  char tmp_buf[512];
  align_line(i_code);
  sprintf( tmp_buf, "#pragma omp parallel\n");
}

void emit_parallel_region(loop_code *i_code) {
  char tmp_buf[512];
  align_line(i_code);
  sprintf( tmp_buf, "#pragma omp parallel\n");
  add_buf_to_code(i_code, tmp_buf);
  align_line(i_code);
  sprintf( tmp_buf, "{\n");
  add_buf_to_code(i_code, tmp_buf);
  increase_nest_level(i_code);
  return;
}

void close_parallel_region(loop_code *i_code) {
  char tmp_buf[512];
  decrease_nest_level(i_code);
  align_line(i_code);
  sprintf( tmp_buf, "}\n");
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_header(loop_code *i_code, loop_param_t *i_loop_param) {
  char tmp_buf[512];
  char str_idx[512];
  char str_start[512];
  char str_end[512];
  char str_step[512];
  
  if ( strcmp(i_loop_param->idx_name, "") == 0 ) {
    sprintf(str_idx, "i%d", i_loop_param->idx_id);
  } else {
    sprintf(str_idx, "%s", i_loop_param->idx_name); 
  }

  if ( strcmp(i_loop_param->start_var_name, "") == 0 ) {
    sprintf(str_start, "%ld", i_loop_param->start);
  } else {
    sprintf(str_start, "%s", i_loop_param->start_var_name); 
  }

  if ( strcmp(i_loop_param->end_var_name, "") == 0 ) {
    sprintf(str_end, "%ld", i_loop_param->end);
  } else {
    sprintf(str_end, "%s", i_loop_param->end_var_name); 
  }

  if ( strcmp(i_loop_param->step_var_name, "") == 0 ) {
    sprintf(str_step, "%ld", i_loop_param->step);
  } else {
    sprintf(str_step, "%s", i_loop_param->step_var_name); 
  }

  align_line(i_code);
  sprintf( tmp_buf, "for (int %s = %s; %s < %s; %s += %s) {\n", str_idx, str_start, str_idx, str_end, str_idx, str_step);
  add_buf_to_code(i_code, tmp_buf);
  increase_nest_level(i_code);

  return;
}

void emit_func_signature(loop_code *i_code, char *spec_func_name, char *body_func_name, char *init_func_name, char *term_func_name) {
  char tmp_buf[512];
  int i;
  align_line(i_code);
  sprintf(tmp_buf, "extern \"C\" void par_nested_loops(loop_rt_spec_t *%s, std::function<void(int *)> %s, std::function<void()> %s, std::function<void()> %s) {\n", spec_func_name, body_func_name, init_func_name, term_func_name );
  add_buf_to_code(i_code, tmp_buf);
  increase_nest_level(i_code);
}

void emit_func_termination(loop_code *i_code) {
  char tmp_buf[512];
  decrease_nest_level(i_code);
  align_line(i_code);
  sprintf(tmp_buf,"}\n");
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_void_function(loop_code *i_code, char *func_name) {
  char tmp_buf[512];
  align_line(i_code);
  sprintf(tmp_buf,"%s();\n", func_name);
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_body(loop_code *i_code, char *body_func_name) {
  char tmp_buf[512];
  int i;
  align_line(i_code);
  sprintf(tmp_buf, "int idx[%d];\n", i_code->n_logical_loops);
  add_buf_to_code(i_code, tmp_buf);
  /* Here we set the idx array to be used by function called */
  for (i = 0; i < i_code->n_logical_loops; i++) {
    char str_idx[64];
    sprintf(str_idx, "%c%d", 'a'+i, i_code->occurence_map['a'+i]-1);
    align_line(i_code);
    sprintf(tmp_buf, "idx[%d] = %s;\n", i, str_idx);
    add_buf_to_code(i_code, tmp_buf);
  }
  align_line(i_code);
  sprintf(tmp_buf, "%s(idx);\n", body_func_name);
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_termination(loop_code *i_code) {
  char tmp_buf[512];
  decrease_nest_level(i_code);
  align_line(i_code);
  sprintf(tmp_buf,"}\n");
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_barrier(loop_code *i_code) {
  char tmp_buf[512];
  align_line(i_code);
  sprintf(tmp_buf,"#pragma omp barrier\n");
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void set_loop_param( loop_param_t *io_param, const char *idx_name, const char *s_name, const char *e_name, const char *step_name, int pos) {
  io_param->pos_in_loopnest = pos;
  sprintf(io_param->idx_name, "%s", idx_name);
  sprintf(io_param->start_var_name, "%s", s_name);
  sprintf(io_param->end_var_name, "%s", e_name);
  sprintf(io_param->step_var_name, "%s", step_name);
  return;
}

int is_simple_char(char cur) {
  int result = 0;
  if ( (cur >= 'a' && cur <= 'z') ||
       (cur >= 'A' && cur <= 'Z') ||
       (cur == '|') ) {
    result = 1;
  }
  return result;
}

void parse_jit_info(char *jit_info_str, loop_param_t *loop_param) {
  char cur_token[512];
  char token_start[512];
  char token_end[512];
  char token_step[512];
  char token_bs[512];
  int i = 0;
  int j = 0;
  int token_id = 0;
  char *bs_str;
  int bs_index = 0;

  /* First extract the BS token */
  while (jit_info_str[i] != '(') {
    i++;
  }
  jit_info_str[i] = '\0';
  i++;
  while(jit_info_str[i] != ')') {
    token_bs[j] = jit_info_str[i];
    j++;
    i++;
  }
  token_bs[j] = '\0';

  /* Now extract rest token */
  i = 0;
  j = 0;
  while (jit_info_str[i] != '\0') {
    if (jit_info_str[i] == ',') {
      if (i == 0) {
        /* Empty token */
        if (token_id == 0) {
          sprintf(token_start, "");
        } else if (token_id == 1) {
          sprintf(token_end, "");
        } else if (token_id == 2) {
          sprintf(token_step, ""); 
        }
        token_id++;
      } else if (jit_info_str[i-1] == ',') {
        /* Empty token */
        if (token_id == 0) {
          sprintf(token_start, "");
        } else if (token_id == 1) {
          sprintf(token_end, "");
        } else if (token_id == 2) {
          sprintf(token_step, ""); 
        }
        token_id++;  
      } else {
        /* Finalize current token */
        cur_token[j] = '\0';
        j = 0;
        if (token_id == 0) {
          sprintf(token_start, "%s", cur_token);
        } else if (token_id == 1) {
          sprintf(token_end, "%s", cur_token);
        } else if (token_id == 2) {
          sprintf(token_step, "%s", cur_token); 
        }
        token_id++;
      }
    } else {
      cur_token[j] = jit_info_str[i];
      j++;
    } 
    i++;
  }
  
  /* Now based on token parse info... */
  if (strlen(token_start) > 0) {
    loop_param->jit_start = 1;
    loop_param->start= atoi(token_start);
  }

  if (strlen(token_end) > 0) {
    loop_param->jit_end = 1;
    loop_param->end= atoi(token_end);
  }

  if (strlen(token_step) > 0) {
    loop_param->jit_step = 1;
    loop_param->step = atoi(token_step);
  }

  if (strlen(token_bs) > 0) { 
    bs_str = strtok(token_bs, ",");
    bs_index = 0;
    while (bs_str != NULL) {
      loop_param->jit_block_sizes = 1;
      loop_param->block_size[bs_index] = atoi(bs_str);
      bs_index++;
      bs_str = strtok(NULL, ",");
    }
  }
}

void extract_jit_info(const char *in_desc, char *out_desc, loop_param_t *loop_params) {
  int i = 0, k = 0;
  char jit_params_str[512];
  char loop_id;

  while (i < strlen(in_desc)) {
    char cur = in_desc[i];
    if (is_simple_char(cur)) {
      out_desc[k] = cur;
      k++;
      i++;
      if (cur != '|') {
        loop_id = tolower(cur);
      }
    } else {
      /* Start reading specs string [ .. ] */
      if (cur == '[') {
        int j = 0;
        while ( cur != ']' ) {
          i++;
          cur = in_desc[i];
          if (cur != ']') {
            jit_params_str[j] = cur;
            j++;
          } else {
            i++;
          }
        }
        jit_params_str[j] = '\0';
        parse_jit_info(jit_params_str, &loop_params[loop_id-'a']);
      } 
    }
  }
  out_desc[k] = '\0';
}

void loop_generator( FILE *fp_out, const char *_loop_nest_desc_extended ) {
  char body_func_name[64] = "body_func";
  char init_func_name[64] = "init_func";
  char term_func_name[64] = "term_func";
  char spec_func_name[64] = "loop_rt_spec";
  char loop_map[256];
  char occurence_map[256];
  loop_code l_code;
  char *result_code ;
  loop_param_t loop_params[256], cur_loop, loop_params_map[256];
  int n_loops, n_logical_loops, i, k, have_emitted_parallel_for = 0, n_parallel_loops = 0;
  char loop_nest_desc[256];
  char barrier_positions[256];
  int jit_loop_spec = 0;
  char loop_nest_desc_extended[MAX_LOOP_DESC_SIZE];
  // Warn if input descriptor is too long
  if (strlen(_loop_nest_desc_extended) >= MAX_LOOP_DESC_SIZE) {
    fprintf(stderr, "Warning: loop descriptor string is too long and may be truncated.\n");
  }

  /* Check if we have to jit the loop specs  */
  for (i=0; i < strlen(_loop_nest_desc_extended); i++) {
    if (_loop_nest_desc_extended[i] == '[') {
      jit_loop_spec = 1;
      break;
    }
  }
  l_code.jit_loop_spec = jit_loop_spec;
  
  memset(loop_params_map, 0, 256 * sizeof(loop_param_t));
  if (jit_loop_spec > 0) {
    extract_jit_info(_loop_nest_desc_extended, loop_nest_desc_extended, loop_params_map);
  } else {
    strcpy(loop_nest_desc_extended, _loop_nest_desc_extended);
  }

  /* Cleanup input descriptor to exclude barriers */
  k = 0;
  memset(barrier_positions, 0, 256);
  for (i=0; i < strlen(loop_nest_desc_extended); i++) {
    if (loop_nest_desc_extended[i] == '|') {
      if (k-1 >= 0) {
        barrier_positions[k-1] = 1;
      }
    } else {
      loop_nest_desc[k] = loop_nest_desc_extended[i];
      k++;
    }
  }
  loop_nest_desc[k] = '\0';

  n_loops = strlen(loop_nest_desc);
  result_code = (char*) malloc(MAX_CODE_SIZE *sizeof(char));

  l_code.buf = result_code;
  l_code.cur_nest_level = 0;
  l_code.n_loops = n_loops;
  l_code.loop_params = loop_params;
  l_code.cur_pos = 0;

  /* Find number of parallel loops */
  for (i = 0; i < n_loops; i++) {
    if (tolower(loop_nest_desc[i]) != loop_nest_desc[i]) {
      n_parallel_loops++;
    }
  }

  /* Count how many times each loop occurs (lower case and upper case are equivalent for that matter) */
  memset(loop_map, 0, 256* sizeof(char));
  for (i = 0; i < n_loops; i++) {
    loop_map[tolower(loop_nest_desc[i])]++;
  }

  /* Set up loop properties */
  memset(occurence_map, 0, 256*sizeof(char));
  for (i = 0; i < n_loops; i++) {
    int is_blocked = (loop_map[tolower(loop_nest_desc[i])] > 1) ? 1 : 0;
    int is_parallelizable = (tolower(loop_nest_desc[i]) != loop_nest_desc[i]) ? 1 : 0;
    int occurence_id, is_blocked_outer;
    char idx_name[16];
    char spec_array_name[512];
    char start_var_name[512];
    char end_var_name[512];
    char step_var_name[512];
    int loop_abs_index = tolower(loop_nest_desc[i]) - 'a';

    occurence_id = occurence_map[tolower(loop_nest_desc[i])];
    is_blocked_outer = (occurence_id == 0) ? 1 : 0;
    occurence_map[tolower(loop_nest_desc[i])]++;

    sprintf(spec_array_name, "%s", spec_func_name);

    sprintf(idx_name, "%c%d", tolower(loop_nest_desc[i]), occurence_id);

    if (occurence_id == 0) {
      if (loop_params_map[loop_abs_index].jit_start > 0) {
        sprintf(start_var_name, "%ld", loop_params_map[loop_abs_index].start); 
      } else {
        sprintf(start_var_name, "%s[%d].start", spec_array_name, loop_abs_index);
      }
    } else {
      sprintf(start_var_name, "%c%d", tolower(loop_nest_desc[i]), occurence_id-1);
    }

    if (occurence_id == 0) {
      if (loop_params_map[loop_abs_index].jit_end > 0) {
        sprintf(end_var_name, "%ld", loop_params_map[loop_abs_index].end); 
      } else {
        sprintf(end_var_name, "%s[%d].end", spec_array_name, loop_abs_index);
      }
    } else {
      if (loop_params_map[loop_abs_index].jit_block_sizes > 0) {
        sprintf(end_var_name, "%c%d + %ld", tolower(loop_nest_desc[i]), occurence_id-1, loop_params_map[loop_abs_index].block_size[occurence_id-1]);
      } else {
        sprintf(end_var_name, "%c%d + %s[%d].block_size[%d]", tolower(loop_nest_desc[i]), occurence_id-1, spec_array_name, loop_abs_index, occurence_id-1);
      }
    }

    if (is_blocked) {
      if (occurence_id == loop_map[tolower(loop_nest_desc[i])]-1) {
        if (loop_params_map[loop_abs_index].jit_step > 0) {
          sprintf(step_var_name, "%ld", loop_params_map[loop_abs_index].step);
        } else {
          sprintf(step_var_name, "%s[%d].step", spec_array_name, loop_abs_index);
        }
      } else {
        if (loop_params_map[loop_abs_index].jit_block_sizes > 0) {
          sprintf(step_var_name, "%ld", loop_params_map[loop_abs_index].block_size[occurence_id]);
        } else {
          sprintf(step_var_name, "%s[%d].block_size[%d]", spec_array_name, loop_abs_index, occurence_id);
        }
      }
    } else {
      if (loop_params_map[loop_abs_index].jit_step > 0) {
        sprintf(step_var_name, "%ld", loop_params_map[loop_abs_index].step);
      } else {
        sprintf(step_var_name, "%s[%d].step", spec_array_name, loop_abs_index);
      }
    }
    
    set_loop_param( &loop_params[i], idx_name, start_var_name, end_var_name, step_var_name, i);
    loop_params[i].is_parallelizable = is_parallelizable;
    loop_params[i].is_blocked = is_blocked;
    loop_params[i].is_blocked_outer = is_blocked_outer;
  }

  /* Setup number of logical loops and the ocurence map */
  n_logical_loops = 0;
  for (i = 0; i < 256; i++) {
    if (occurence_map[i] > 0) {
      n_logical_loops++;
    }
  }
  l_code.n_logical_loops = n_logical_loops;

  memcpy(&l_code.occurence_map[0], occurence_map, 256);

  /* Emit function signature  */
  emit_func_signature(&l_code, spec_func_name, body_func_name, init_func_name, term_func_name);

  /* Emit loop function header */
  if (n_parallel_loops > 0) {
    emit_parallel_region(&l_code);
  }

  /* Emit init function */
  emit_void_function(&l_code, init_func_name);

  for (i = 0; i < n_loops; i++){
    cur_loop = loop_params[i];
    /* Emit paralle for if need be*/
    if ((cur_loop.is_parallelizable == 1) && (have_emitted_parallel_for == 0)) {
      int collapse_level = 1;
      int j = i+1;
      int is_parallel = 1;
      while ((is_parallel > 0) && (j < n_loops)) {
        loop_param_t tmp_loop = loop_params[j];
        if (tmp_loop.is_parallelizable > 0) {
          collapse_level++;
          j++;
        } else {
          is_parallel = 0;
        }
      }
      emit_parallel_for(&l_code, collapse_level);
      have_emitted_parallel_for = 1;
    }
    emit_loop_header(&l_code, &cur_loop);  
  }

  emit_loop_body(&l_code, body_func_name);

  for (i = n_loops-1; i >= 0; i--){
    emit_loop_termination(&l_code);
    if (barrier_positions[i] > 0) {
      emit_barrier(&l_code);
    }
  }

  /* Emit term function */
  emit_void_function(&l_code, term_func_name);

  if (n_parallel_loops > 0) {
    close_parallel_region(&l_code);
  }
  
  emit_func_termination(&l_code);
  
  fprintf(fp_out, "%s", result_code);
  fprintf(stderr, "%s", result_code);

  if (result_code) free(result_code);

  return;
}


/* Autotuner input *
 *
 *  loopVarName_blockTimes_brgemmDimType=M/N/K/R/-,
 *
 * */

void permute(FILE *fout, int i, string s)
{
  int j;
  char prev = '-';
  if (i == (s.length() - 1)) {
    /* Check that we don't have back to back blocked same charcters */
    char print_str[256];
    char print_str2[256];
    int k = 0;
    int pos = 0;
    print_str[pos] = s[k];
    pos++;
    for (k = 1; k < s.length(); k++) {
      if (s[k] != s[k-1]) {
        print_str[pos] = s[k];
        pos++;
      }
    }
    print_str[pos] = '\0';

    /*Exclude loops of Type X....Z since effectively they are not collapsed... */
    for (j = 0; j < strlen(print_str); j++) {
      if (print_str[j] >= 'A' && print_str[j] <= 'Z') {
        int z = 0;
        for (z = j + 2/*n_parallelizable*/; z < strlen(print_str); z++) {
          if (print_str[z] >= 'A' && print_str[z] <= 'Z') {
            print_str[z] = tolower(print_str[z]);   
          }
        }
        break;
      }
    }

    sprintf(print_str2,"%s", print_str);
    k = 0;
    pos = 0;
    print_str[pos] = print_str2[k];
    pos++;
    for (k = 1; k < strlen(print_str2); k++) {
      if (print_str2[k] != print_str2[k-1]) {
        print_str[pos] = print_str2[k];
        pos++;
      }
    }
    print_str[pos] = '\0';

#if 0
    if (print_str[0] != 'a') {
      return;
    }
#endif

    string _s(print_str);
    fprintf(fout,"%s\n", print_str);
    //cout << _s << endl;
    return;
  }
  for (j = i; j < s.length(); j++) {
    string temp = s;
    if (j > i && temp[i] == temp[j]) {
      continue;
    }
    if (prev != '-' && prev == s[j]) {
      continue;
    }
    swap(temp[i], temp[j]);
    prev = s[j];
    permute(fout, i + 1, temp);
  }
}

void get_loop_properties(FILE *my_permutes, char *inp_str) {
  char tmp[256];
  loop_metadata_t loop_metadata_map[256];
  loop_metadata_t tmp_map_entry;
  char tmp_in[256];
  char *token, *token_in;
  char *inp_rest = inp_str;
  char *loop_rest;
  char loop_nest_desc[256];
  char loop_id;
  char type_id;
  int n_loops = 0;
  int cur_loops;
  int i = 0;
  int n_parallel_loop_combos = 1;
  int n_vars = 0;
  int n_parallelizable = 0;
  int parallel_id_to_char_id[32];

  token = strtok_r(inp_rest, ",", &inp_rest);
  while (token != NULL) {
    sprintf(tmp, "%s", token);
    loop_rest = tmp;
    token_in = strtok_r(loop_rest, "_", &loop_rest);
    sprintf(tmp_in, "%s", token_in);
    loop_id = tolower(tmp_in[0]); 
    //printf("name is %s ", tmp_in);
    token_in = strtok_r(loop_rest, "_", &loop_rest);
    sprintf(tmp_in, "%s", token_in);
    cur_loops = atoi(tmp_in)+1;
    //printf("blocked %s times ", tmp_in);
    token_in = strtok_r(loop_rest, "_", &loop_rest);
    sprintf(tmp_in, "%s", token_in);
    type_id = tmp_in[0];
    //printf("and %s type\n", tmp_in);
    tmp_map_entry.name = loop_id;
    tmp_map_entry.n_blocked_times = cur_loops;
    tmp_map_entry.type = type_id;
    tmp_map_entry.is_parallelizable = (tolower(type_id) == 'm' || tolower(type_id) == 'n') ? 1 : 0;
    tmp_map_entry.pos_first_occ = n_loops;
    loop_metadata_map[loop_id] = tmp_map_entry;
    n_vars++;
    if (tmp_map_entry.is_parallelizable) {
      parallel_id_to_char_id[n_parallelizable] = loop_id;
      n_parallelizable++;
      n_parallel_loop_combos *= 2;
    }
    /* Generate "sorted" string given #of blocked times */
    for (i = 0; i < cur_loops; i++) {
      loop_nest_desc[n_loops] = loop_id;
      n_loops++;
    }
    token = strtok_r(inp_rest, ",", &inp_rest);
  }
  loop_nest_desc[n_loops] = '\0';

  /* We create all unique permutations */
  //string s(loop_nest_desc);
  //permute(0, s);

  /* For each permutation we can decide if we parallelize a loop or not */
  /* Each loop index can be parallelized once if it is M or N type for now... */
  if (n_parallelizable > 0) {
    for ( i = 1; i < n_parallel_loop_combos; i++) {
      char new_par_loop_combo[256];
      int cur_combo_int = i;
      int j;
      sprintf(new_par_loop_combo, "%s", loop_nest_desc);
      /* Extract j-th bit value from combo int */
      for (j = 0; j < n_parallelizable; j++) {
        int mask = 1 << j;
        mask = mask & cur_combo_int;
        mask = mask >> j;
        if (mask) {
          char char_id = parallel_id_to_char_id[j];
          int pos = loop_metadata_map[char_id].pos_first_occ;
          new_par_loop_combo[pos] = toupper(char_id);
        }      
      }
      //printf("Loop Combo is:  %s\n", new_par_loop_combo);
      /*Exclude loops of Type X....Z since effectively they are not collapsed... */
#if 0
      for (j = 0; j < strlen(new_par_loop_combo); j++) {
        if (new_par_loop_combo[j] >= 'A' && new_par_loop_combo[j] <= 'Z') {
          int z = 0;
          for (z = j + n_parallelizable; z < strlen(new_par_loop_combo); z++) {
            if (new_par_loop_combo[z] >= 'A' && new_par_loop_combo[z] <= 'Z') {
              new_par_loop_combo[z] = tolower(new_par_loop_combo[z]);   
            }
          }
          break;
        }
      }
#endif
      string s(new_par_loop_combo);
      permute(my_permutes, 0, s);
    }
  } else {
    string s(loop_nest_desc);
    permute(my_permutes, 0, s);
  }
}

#if 0
int main(int argc, char **argv) {
  FILE *my_prog;
  char loop_nest_desc[64] = "dcdABa";
  char loop_nest_tuner[64] = "d_0_K,a_12_M,c_7_-";

  if (argc > 1) {
    sprintf(loop_nest_desc, "%s", argv[1]);
  }
  if (argc > 2) {
    sprintf(loop_nest_tuner, "%s", argv[2]);
  }

  my_prog = fopen("nested_loops.c", "w");

  loop_generator( my_prog, loop_nest_desc);

  fclose(my_prog);

  get_loop_properties(loop_nest_tuner);

  return 0;
}
#else
int main(int argc, char **argv) {
  FILE *my_permutes = fopen("loop_permutes.txt", "w");
  char loop_nest_name[64] = "basic_loop";
  char loop_nest_tuner[64] = "a_0_M,b_0_N,c_0_K";
  char cmd[256];
  if (argc > 1) {
    sprintf(loop_nest_name, "%s", argv[1]);
  }
  if (argc > 2) {
    sprintf(loop_nest_tuner, "%s", argv[2]);
  }

  get_loop_properties(my_permutes, loop_nest_tuner);
  fclose(my_permutes);
  sprintf(cmd,"awk '!seen[$0]++' loop_permutes.txt > %s_bench_configs.txt", loop_nest_name);
  system(cmd);
  
  return 0;
}
#endif


