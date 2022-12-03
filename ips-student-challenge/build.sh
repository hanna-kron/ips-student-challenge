#!/bin/bash
# build.sh - Build all my code for the IPS Student Challenge
# Author: hanna
#

mkdir -p build/

# Configuration options

cuda_enabled=1
debug_build=0

# Compilation stuff:

warnings="-Wall -Werror -Wuninitialized -Wno-missing-braces -Wno-unused-variable -Wno-unused-function -Wno-error=deprecated-declarations -Wno-sign-compare -Wno-write-strings -Wno-unused-local-typedefs -Wno-unused-value -Wno-self-assign"
switches="-DCUDA_ENABLED=$cuda_enabled -DDEBUG_BUILD=$debug_build"
flags="-g3 -I.. -lm -lpthread -ldl -mssse3 -mavx -maes -mbmi2 $warnings $switches"

if [[ $cuda_enabled -eq 1 ]] ; then
  echo "compling cuda program"
  # NOTE(hanna - 2022-12-02): Enabling relocatable-device-code leads to CUDA randomly freezing (and creating an unkillable process) on my desktop upon the first CUDA call.
  # Unfortunate because that setting is neccesary for dynamic parallelism.
  nvcc --compiler-options -fPIC -shared kernel.cu -o build/kernel.so -I.. $switches --compiler-options -g3 --compiler-options "-Wall -Wno-unused-function -Wno-sign-compare"
  if [[ ! $? -eq 0 ]] ; then
    echo "Failed to compile CUDA program"
    exit 1
  fi
fi

echo "compiling debug build"
gcc main.c -o build/debug-ips -O0 $flags
if [[ ! $? -eq 0 ]] ; then
  echo "failed to compile debug build"
  exit 1
fi

echo "compiling release build"
gcc main.c -o build/release-ips -O3 $flags
if [[ ! $? -eq 0 ]] ; then
  echo "failed to compile release build"
  exit 1
fi

echo "compling gen-positions.c"
gcc gen-positions.c -o build/gen-positions -O0 $flags
if [[ ! $? -eq 0 ]] ; then
  echo "failed to compile gen-positions.c"
  exit 1
fi

echo "running build"
build/release-ips

