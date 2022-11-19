# Usage:
# <to be added>

# Help message echo
# <to be added>

# Input arguments checking
# <to be added>

prec=$1
layer_id=$2
mb=$3
nthreads=$4

niters=1000

echo "setup_env_... script called with: prec = $prec layer_id = $layer_id mb = $mb nthreads = $nthreads"

# A utility used for design #2 of setting the metadata (see below)
# Can set everything but currently sets everything except use_bf16 and mb
setenv_conv_fwd ()
{
  #
  # ${1}  = H
  # ...

  # For debugging purposes, will be removed in the final version if this stays at all
  echo "Params:"
  echo "1 = ${1}"
  echo "2 = ${2}"
  echo "3 = ${3}"
  echo "4 = ${4}"
  echo "5 = ${5}"
  echo "6 = ${6}"
  echo "7 = ${7}"
  echo "8 = ${8}"
  echo "9 = ${9}"
  echo "10 = ${10}"
  echo "11 = ${11}"
  echo "12 = ${12}"
  echo "13 = ${13}"
  echo "14 = ${14}"

  export tst_h=${1}
  export tst_w=${2}
  export tst_c=${3}
  export tst_k=${4}
  export tst_r=${5}
  export tst_s=${6}
  export tst_u=${7}
  export tst_v=${8}
  export tst_ph=${9}
  export tst_pw=${10}
  export tst_bc=${11}
  export tst_bk=${12}
  export tst_tunestring=${13}
  export tst_hblock=${14}
  export tst_wblock=${15}
  export tst_cblock=${16}
  export tst_kblock=${17}
  export tst_hingemm=${18}
  export tst_packinput=${19}
  export tst_niters=${20}
  export tst_logicalpad=${21}
}

if [[ "$prec" == "bf16" ]]; then
  export tst_usebf16=1
  if [[ "$mb" == "56" ]]; then
    export tst_mb=$mb
    if [[ "$nthreads" == "56" ]]; then
      export tst_nthreads=$nthreads
      if [[ "$layer_id" == "rn50_conv1_fwd" ]]; then
        # design #1, exporting everything line by line (code bloat?)
        export tst_h=224
        export tst_w=224
        export tst_c=4
        export tst_k=64
        export tst_r=7
        export tst_s=7
        export tst_u=2
        export tst_v=2
        export tst_ph=3
        export tst_pw=3
        export tst_bc=4
        export tst_bk=64
        export tst_tunestring=Afgbdced
        export tst_hblock=1
        export tst_wblock=2
        export tst_cblock=1
        export tst_kblock=1
        export tst_hingemm=1
        export tst_packinput=0
        export tst_niters=1000
        export tst_logicalpad=1
        #export ...

        # design #2: using once-written function for conv_fwd (also works)
        #setenv_conv_fwd  224 224 4 64 7 7 2 2 3 3  4 64 Afgbdced 1 2 1 1 1 0 $niters 1
      fi
    fi
  fi
fi

# Call example (two options):
: '
# Preamble (done once)
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=/swtools/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD

# Checking for environment clash (checking that none of the parameters tst_<...> are defined already, to avoid weird collisions)
# <to be added>

# Calling a single case, design A
source setup_env_convfwd_spr_mb56_thr56.sh bf16 rn50_conv1_fwd 56 56
USE_BF16=$tst_usebf16 OMP_NUM_THREADS=$tst_nthreads ./conv_fwd $tst_tunestring $tst_mb $tst_h $tst_w $tst_c $tst_k $tst_r $tst_s $tst_u $tst_v $tst_ph $tst_pw $tst_bc $tst_bk $tst_hblock $tst_wblock $tst_cblock $tst_kblock $tst_hingemm $tst_packinput $tst_niters $tst_logicalpad

# Calling a single case, design B (not implemented)
source setup_env_convfwd_spr_mb56_thr56.sh bf16 rn50_conv1_fwd 56 56
 ./conv_fwd # no parameters, we read them from the environment variables tst_<varname>
'

# design B (not implemented)
# ./conv_fwd # no parameters, we read them from the environment variables tst_<varname>
return

