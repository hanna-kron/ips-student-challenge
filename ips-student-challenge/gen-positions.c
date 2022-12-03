/*
gen-positions.c - Generate randomly distributed points in 3D space
Author: hanna
Created: 2022-11-21

*/

#include "shared/hanna_util.h"

int main(int argc, char **argv){
  PCG_State rng = pcg_create_with_os_entropy();

  Allocator *allocator = heap_allocator_make(NULL);

  for(I64 N = 1; N <= 1<<22; N <<= 1){
    StringBuilder out = sb_create(allocator);

    F64 rho = 16000/(100); // points per unit volume

    F32 d0 = pow(8.0 * N / rho, 1.0 / 3.0);

    fiz(N){
      F32 x = pcg_random_f32(&rng, -d0, d0);
      F32 y = pcg_random_f32(&rng, -d0, d0);
      F32 z = pcg_random_f32(&rng, -d0, d0);
      sb_printf(&out, "%.6f %.6f %.6f\n", x, y, z);
    }

    String path = allocator_push_printf(allocator, "data/positions-%I64i.xyz", N);
    bool status = dump_string_to_file(path, sb_to_string(&out, allocator));
    if(!status){
      panic("Unable to write file %.*s\n", StrFormatArg(path));
    }
  }

  return 0;
}
