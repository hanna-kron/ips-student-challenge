/*
common.h - Code shared between main.c and kernel.cu. Also some configuration options.
Author: hanna
Created: 2022-11-21

*/

//  ===============================================
// Hi there!, here we have the configuration options:

#define USE_SPALL 0 // Spall is a nice profiling tool
// #define ENABLE_CUDA is configured in ./build.sh
// #define DEBUG_BUILD is configured in ./build.sh

// Select between the three approaches I kept for the final version of this project:
#define APPROACH_kd_tree     1
#define APPROACH_range_tree  2
#define APPROACH_gpu_grid    3

#define APPROACH APPROACH_kd_tree // <--- Here you select which approach to use

#if USE_SPALL
#define SPALL_IMPLEMENTATION
#include "3rdparty/spall.h"
#else
typedef struct SpallProfile{ /* nothing here */ }SpallProfile;
#endif

#include <x86intrin.h>

//
// GLOBAL CONSTANTS
//

// NOTE(hanna): Originally I used 0.5f here without realizing that it was wrong. Thus
// all the code was optimized for that case. Luckily, however, it seems like it still works
// fine for the lower relative point density.
#define CRITICAL_DISTANCE (0.05f)

typedef struct Points Points;
struct Points{
  union{
    struct{
      F32 *x, *y, *z;
    };
    F32 *e[3];
  };
  U32 count;
};

static Points points_subarray(Points points, U32 ps_begin, U32 ps_end){
  Points result = {0};
  assert(ps_begin <= ps_end);
  assert(ps_end <= points.count);
  result.count = ps_end - ps_begin;
  result.x = points.x + ps_begin;
  result.y = points.y + ps_begin;
  result.z = points.z + ps_begin;
  return result;
}

//
// THREAD CONTEXT
//

typedef struct ThreadContext ThreadContext;
struct ThreadContext{
  #if USE_SPALL
  U32 spall_tid;
  SpallProfile *spall;
  SpallBuffer spall_buffer;
  #endif
};
#ifdef __cplusplus
static thread_local ThreadContext *global_thread_context;
#else
static _Thread_local ThreadContext *global_thread_context;
#endif

static AtomicU32 _global_tid;

static void thread_context_init(SpallProfile *spall){
  assert(!global_thread_context);
  global_thread_context = (ThreadContext*)calloc(1, sizeof(ThreadContext));
  if(!global_thread_context){
    panic("calloc failed");
  }

  #if USE_SPALL
  global_thread_context->spall_tid = atomic_add_u32(&_global_tid, 1);

  global_thread_context->spall = spall;

  global_thread_context->spall_buffer.length = MEGABYTES(1);
  global_thread_context->spall_buffer.data = os_alloc_pages_commit(global_thread_context->spall_buffer.length);
  if(!global_thread_context->spall_buffer.data){
    panic("Unable to aquire memory for SPALL buffer");
  }
  if(!SpallBufferInit(spall, &global_thread_context->spall_buffer)){
    panic("SpallBufferInit failed");
  }
  #endif
}

static void thread_context_destroy(){
  assert(global_thread_context);

#if USE_SPALL
  if(!SpallBufferQuit(global_thread_context->spall, &global_thread_context->spall_buffer)){
    panic("SpallBufferQuit failed");
  }
  os_free_pages(global_thread_context->spall_buffer.data, global_thread_context->spall_buffer.length);
#endif

  free(global_thread_context); global_thread_context = NULL;
}

//
// PROFILING STUFFS WITH SPALL
//

#if USE_SPALL

static double GET_CURRENT_TIME_US_FOR_SPALL(){
  return __rdtsc();
}

static void PROFILE_beginx(String name){
  SpallTraceBeginLenTid(global_thread_context->spall, &global_thread_context->spall_buffer,
                        (char*)name.data, name.size,
                        global_thread_context->spall_tid,
                        GET_CURRENT_TIME_US_FOR_SPALL());
}
#define PROFILE_begin(...) PROFILE_beginx(LIT_STR(__VA_ARGS__))


static void PROFILE_end(){
  SpallTraceEndTid(global_thread_context->spall, &global_thread_context->spall_buffer,
                   global_thread_context->spall_tid,
                   GET_CURRENT_TIME_US_FOR_SPALL());
}

#else

#define PROFILE_begin(...)
#define PROFILE_beginx(...)
#define PROFILE_end(...)

#endif

//
// INTERFACE BETWEEN main.c and kernel.cu
//

#if CUDA_ENABLED

typedef struct GPUContext GPUContext;
struct GPUContext{
  U32 *gpu_count;
};

typedef struct GPU_API GPU_API;
struct GPU_API{
  void (*init)(GPUContext *gpu, ThreadContext *thread_context);

  Points (*upload_points)(Points points);

  void (*count_internal_pairs_async)(GPUContext *gpu, F32 *gpu_xs, F32 *gpu_ys, F32 *gpu_zs, U32 count);
  void (*count_pairs_between_async)(GPUContext *gpu,
                                    F32 *x0, F32 *y0, F32 *z0, U32 count0,
                                    F32 *x1, F32 *y1, F32 *z1, U32 count1);

  U32 (*get_pair_count)(GPUContext *gpu);

  U32 (*grid_approach)(Points gpu_points, Rect3 bounds);
};

typedef GPU_API (*GPU_GetAPIProc)();
static GPU_API global_gpu_api;

#endif // CUDA_ENABLED
