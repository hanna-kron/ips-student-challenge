/*
kernel.cu - CUDA code for counting number of pairs of points within a certain distance from eachother.
Created: 2022-11-20
Author: hanna

This is my first CUDA program ever so I have no idea what I am doing.
*/

#include <stddef.h>
#include <stdint.h>
#include "shared/hanna_util.h"

#include "common.h"

// NOTE(hanna - 2022-11-26): This constant has been tuned for optimal performance on my desktop (GTX 1060)
#define CUDA_THREAD_COUNT 128

//
// Routines for brute-force solving of subproblems from CPU land
//

__global__
void KERNEL_count_internal_pairs(U32 *out_count, F32 *xs, F32 *ys, F32 *zs, U32 count){
  U32 index = blockIdx.x * blockDim.x + threadIdx.x;

  // NOTE(hanna - 2022-12-02): The 23 bits of mantissa should be plenty for us to get the correct result here.
  // `index` = 70368744177664 is the first point where this breaks down (given that floating point operations
  // are performed in the same order as my CPU-side testing code). We should thus safely be able to handle `count`
  // being ~8000000. If this was a proper codebase I would actually check the computation on the GPU instead of just
  // on the CPU or mathematically justify it. The reason for choosing F32 instead of F64 is that the F32 version is
  // much faster. On the whole task we save ~300us for the KD variant.
  U32 i = (U32)floorf( 0.5f + 0.5f * sqrtf((F32)(1 + 8 * index)) );
  U32 j = index - i * (i - 1) / 2;

  // NOTE(hanna): We check j < count here to make sure we don't mess up when index is too big.
  if(i < count && j < count && SQUARE(xs[i] - xs[j]) + SQUARE(ys[i] - ys[j]) + SQUARE(zs[i] - zs[j]) < SQUARE(CRITICAL_DISTANCE)){
    atomicAdd(out_count, 1); // NOTE: Using atomicAdd (compared to writing to somethinge else non-atomically) doesn't seem to have an effect on performance
  }
}

// TODO: We should probably try out 2D invocation of this kernel
// ^^^ NOTE(hanna - 2022-12-03): Didn't have time to do this :-(
__global__
void KERNEL_count_pairs_between(U32 *out_count,
                                F32 *x0, F32 *y0, F32 *z0, U32 count0,
                                F32 *x1, F32 *y1, F32 *z1, U32 count1)
{
  U32 index = blockIdx.x * blockDim.x + threadIdx.x;

  U32 i = index % count0;
  U32 j = index / count0;

  if(j < count1 && SQUARE(x0[i] - x1[j]) + SQUARE(y0[i] - y1[j]) + SQUARE(z0[i] - z1[j]) < SQUARE(CRITICAL_DISTANCE)){
    atomicAdd(out_count, 1); // NOTE: Using atomicAdd (compared to writing to somethinge else non-atomically) doesn't seem to have an effect on performance
  }
}

#define CUDA_CHECK(...) \
{ \
  cudaError_t error = (__VA_ARGS__);; \
  if(error != cudaSuccess){ \
    panic("CUDA call failed (" #__VA_ARGS__ "): %s", cudaGetErrorString(error)); \
  } \
}

static void GPU_init(GPUContext *gpu, ThreadContext *thread_context){
  global_thread_context = thread_context;
  CUDA_CHECK(cudaMalloc(&gpu->gpu_count, sizeof(U32)));
  CUDA_CHECK(cudaMemsetAsync(gpu->gpu_count, 0, sizeof(U32)));
}

static Points GPU_alloc_points(U32 count){
  Points result = {0};
  result.count = count;

  CUDA_CHECK(cudaMalloc(&result.x, 3 * count * sizeof(F32)));
  result.y = result.x + count;
  result.z = result.y + count;

  return result;
}

static Points GPU_upload_points(Points ps){
  Points result = GPU_alloc_points(ps.count);

  CUDA_CHECK(cudaMemcpyAsync(result.x, ps.x, ps.count * sizeof(F32), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(result.y, ps.y, ps.count * sizeof(F32), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(result.z, ps.z, ps.count * sizeof(F32), cudaMemcpyHostToDevice));

  return result;
}

static void GPU_count_internal_pairs_async(GPUContext *gpu, F32 *gpu_xs, F32 *gpu_ys, F32 *gpu_zs, U32 point_count){
  U32 block_count = (point_count * (point_count - 1) / 2 + CUDA_THREAD_COUNT - 1) / CUDA_THREAD_COUNT;

  KERNEL_count_internal_pairs<<<block_count, CUDA_THREAD_COUNT, 0, 0>>>(gpu->gpu_count, gpu_xs, gpu_ys, gpu_zs, point_count);
  CUDA_CHECK(cudaGetLastError());
}

static void GPU_count_pairs_between_async(GPUContext *gpu,
                                          F32 *x0, F32 *y0, F32 *z0, U32 count0,
                                          F32 *x1, F32 *y1, F32 *z1, U32 count1)
{
  if(count0 && count1){
    U32 block_count = (count0 * count1 + CUDA_THREAD_COUNT - 1) / CUDA_THREAD_COUNT;

    KERNEL_count_pairs_between<<<block_count, CUDA_THREAD_COUNT, 0, 0>>>(gpu->gpu_count, x0, y0, z0, count0, x1, y1, z1, count1);
    CUDA_CHECK(cudaGetLastError());
  }
}

static U32 GPU_get_pair_count(GPUContext *gpu){
  U32 result;
  CUDA_CHECK(cudaMemcpy(&result, gpu->gpu_count, sizeof(U32), cudaMemcpyDeviceToHost));
  return result;
}

//
// GRID APPROACH ENTIRELY* ON THE GPU
//
// * It could theoretically be implemented entirely on the GPU at least, I don't do everything on the GPU.
//

// NOTE(hanna): CELL_DIM has been fine-tuned for optimal performance on the given data set
//#define CELL_DIM (CRITICAL_DISTANCE * 15) // This is optimal when only counting pairs internal to each grid cell
#define CELL_DIM (CRITICAL_DISTANCE * 30) // This works better when counting path "internal" and "external" pairs.

__device__
static U32 HELPER_calculate_grid_index(F32 x, F32 y, F32 z, V3 origin, V3i grid_dim){
  I32 grid_x = CLAMP(0, floor((x - origin.x) / CELL_DIM), grid_dim.x - 1);
  I32 grid_y = CLAMP(0, floor((y - origin.y) / CELL_DIM), grid_dim.y - 1);
  I32 grid_z = CLAMP(0, floor((z - origin.z) / CELL_DIM), grid_dim.z - 1);

  U32 result = grid_x + grid_y * grid_dim.x + grid_z * grid_dim.x * grid_dim.y;
  return result;
}

__global__
static void KERNEL_grid_count(U32 *cell_counts, V3i grid_dim, V3 origin, F32 *xs, F32 *ys, F32 *zs, U32 point_count){
  U32 index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index < point_count){
    U32 grid_index = HELPER_calculate_grid_index(xs[index], ys[index], zs[index], origin, grid_dim);

    atomicAdd(&cell_counts[grid_index], 1);
  }
}

__global__
static void KERNEL_collect_elements(U32 *cell_counts, U32 *cell_point_indices, V3i grid_dim, V3 origin, F32 *xs, F32 *ys, F32 *zs, U32 point_count, F32 *cell_xs, F32 *cell_ys, F32 *cell_zs){
  U32 index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index < point_count){
    U32 grid_index = HELPER_calculate_grid_index(xs[index], ys[index], zs[index], origin, grid_dim);
    U32 point_index = cell_point_indices[grid_index] + atomicAdd(&cell_counts[grid_index], 1);

    cell_xs[point_index] = xs[index];
    cell_ys[point_index] = ys[index];
    cell_zs[point_index] = zs[index];
  }
}

#if 0
// NOTE(hanna): Unfortunately I had to give up on this approach because CUDA randomly froze when enabling --relocatable-device-code=true :-(
__global__
static void KERNEL_grid_approach(U32 *out_count, U32 *cell_counts, U32 *cell_point_indices, V3i grid_dim, F32 *xs, F32 *ys, F32 *zs){
  U32 grid_x = blockIdx.x * blockDim.x + threadIdx.x;
  U32 grid_y = blockIdx.y * blockDim.y + threadIdx.y;
  U32 grid_z = blockIdx.z * blockDim.z + threadIdx.z;

  if(grid_x < grid_dim.x && grid_y < grid_dim.y && grid_z < grid_dim.z){
    U32 grid_index = grid_x + grid_y * grid_dim.x + grid_z * grid_dim.x * grid_dim.y;
    U32 cell_index = cell_point_indices[grid_index];

    U32 block_count = (cell_counts[grid_index] + CUDA_THREAD_COUNT - 1) / CUDA_THREAD_COUNT;

    KERNEL_count_internal_pairs<<<block_count, CUDA_THREAD_COUNT, 0, 0>>>(
      out_count, xs + cell_index, ys + cell_index, zs + cell_index, cell_counts[grid_index]);
  }
}
#endif

typedef struct Grid Grid;
struct Grid{
  V3i dim; // Number of cells in the grid in each dimension
  Points gpu_cell_points; // Points sorted in cell order (no intra-cell sorting)
  U32 *cell_counts; // Number of points in a given cell
  U32 *cell_point_indices; // Map from cell number to index of first point in `cell_points`

  U32 *gpu_cell_counts;
  U32 *gpu_cell_point_indices;
};

static U32 grid_get_grid_index(Grid *grid, I64 x, I64 y, I64 z){
  assert(0 <= x);
  assert(0 <= y);
  assert(0 <= z);
  assert(x < grid->dim.x);
  assert(y < grid->dim.y);
  assert(z < grid->dim.z);
  U32 result = x + y * grid->dim.x + z * grid->dim.x * grid->dim.y;
  return result;
}

static Points grid_get_cell_points(Grid *grid, I32 x, I32 y, I32 z){
  U32 grid_index = grid_get_grid_index(grid, x, y, z);
  return points_subarray(grid->gpu_cell_points, grid->cell_point_indices[grid_index], grid->cell_point_indices[grid_index] + grid->cell_counts[grid_index]);
}

// `points0` here is the points of the main cell, `points1` inside the routine is the points of the neighbour
// `x, y, z` here is the location of the neighbour.
static void grid_cell_vs_neighbour(Grid *grid, U32 *gpu_out_count, Points points0, I32 x, I32 y, I32 z){
  if(1
    && 0 <= x && x < grid->dim.x
    && 0 <= y && y < grid->dim.y
    && 0 <= z && z < grid->dim.z)
  {
    Points points1 = grid_get_cell_points(grid, x, y, z);

    if(points0.count && points1.count){
      U32 block_count = (points0.count * points1.count + CUDA_THREAD_COUNT - 1) / CUDA_THREAD_COUNT;

      KERNEL_count_pairs_between<<<block_count, CUDA_THREAD_COUNT, 0, 0>>>(
        gpu_out_count,
        points0.x, points0.y, points0.z, points0.count,
        points1.x, points1.y, points1.z, points1.count);
      CUDA_CHECK(cudaGetLastError());
    }
  }
}

static U32 GPU_grid_approach(Points gpu_points, Rect3 bounds){
  PROFILE_begin("GPU_grid_approach");
  Grid grid = {0};

  V3 dim = rect3_dim(bounds);
  grid.dim = { (I32)ceilf(dim.x / CELL_DIM), (I32)ceilf(dim.y / CELL_DIM), (I32)ceilf(dim.z / CELL_DIM) };;

  Allocator *allocator = heap_allocator_make(NULL);

  // fprintf(stderr, "grid dim %d, %d, %d --> %.3fMiB\n", (I32)grid_dim.x, (I32)grid_dim.y, (I32)grid_dim.z, grid_dim.x * grid_dim.y * grid_dim.z * sizeof(U16) / 1024.0 / 1024.0);
  U32 grid_element_count = grid.dim.x * grid.dim.y * grid.dim.z;

  // =============================================
  // STEP 1: Count number of elements in each cell

  PROFILE_begin("step 1");
  CUDA_CHECK(cudaMalloc(&grid.gpu_cell_counts, grid_element_count * sizeof(U32)));
  CUDA_CHECK(cudaMemset(grid.gpu_cell_counts, 0, grid_element_count * sizeof(U32)));

  U32 block_count_points = (gpu_points.count + CUDA_THREAD_COUNT - 1) / CUDA_THREAD_COUNT;
  KERNEL_grid_count<<<block_count_points, CUDA_THREAD_COUNT, 0, 0>>>(
    grid.gpu_cell_counts, grid.dim, bounds.min, gpu_points.x, gpu_points.y, gpu_points.z, gpu_points.count
    );
  CUDA_CHECK(cudaGetLastError());

  grid.cell_counts = allocator_push_items_noclear(allocator, U32, grid_element_count);
  CUDA_CHECK(cudaMemcpy(grid.cell_counts, grid.gpu_cell_counts, grid_element_count * sizeof(U32), cudaMemcpyDeviceToHost));
  PROFILE_end();

  // ======================================================
  // STEP 2: Compute prefix sums (unfortunately on the CPU)
  //                             (but this is very small amounts of data so this is fine)

  PROFILE_begin("step 2");

  grid.cell_point_indices = allocator_push_items_noclear(allocator, U32, grid_element_count);
  grid.cell_point_indices[0] = 0;
  for(I64 i = 1; i < grid_element_count; i += 1){
    grid.cell_point_indices[i] = grid.cell_point_indices[i - 1] + grid.cell_counts[i - 1];
  }

  CUDA_CHECK(cudaMalloc(&grid.gpu_cell_point_indices, grid_element_count * sizeof(U32)));
  CUDA_CHECK(cudaMemcpy(grid.gpu_cell_point_indices, grid.cell_point_indices, grid_element_count * sizeof(U32), cudaMemcpyHostToDevice));

  PROFILE_end();

  // ===========================================
  // STEP 3: Collect points into the right cells

  PROFILE_begin("step 3");

  CUDA_CHECK(cudaMemset(grid.gpu_cell_counts, 0, grid_element_count * sizeof(U32)));

  grid.gpu_cell_points = GPU_alloc_points(gpu_points.count);

  KERNEL_collect_elements<<<block_count_points, CUDA_THREAD_COUNT, 0, 0>>>(
    grid.gpu_cell_counts, grid.gpu_cell_point_indices, grid.dim, bounds.min,
    gpu_points.x, gpu_points.y, gpu_points.z, gpu_points.count,
    grid.gpu_cell_points.x, grid.gpu_cell_points.y, grid.gpu_cell_points.z);
  CUDA_CHECK(cudaGetLastError());

  PROFILE_end();

  // ==============================
  // STEP 4: Do the actual counting

  PROFILE_begin("step 4");

#if 0
  U32 result = 0;
  {
    U32 thread_count = 4;

    dim3 block_count_grid = dim3((grid_dim.x + thread_count - 1) / thread_count,
                                 (grid_dim.y + thread_count - 1) / thread_count,
                                 (grid_dim.z + thread_count - 1) / thread_count);
    dim3 thread_count_dim3 = dim3(thread_count, thread_count, thread_count);

    U32 *gpu_pair_count;
    CUDA_CHECK(cudaMalloc(&gpu_pair_count, sizeof(U32)));
    CUDA_CHECK(cudaMemset(gpu_pair_count, 0, sizeof(U32)));

    KERNEL_grid_approach<<<block_count_grid, thread_count_dim3, 0, 0>>>(
      gpu_pair_count, gpu_cell_counts, gpu_cell_point_indices,
      grid_dim,
      gpu_points_cells.x, gpu_points_cells.y, gpu_points_cells.z);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(&result, gpu_pair_count, sizeof(U32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gpu_pair_count));
  }
#else
  U32 result = 0;
  {
    U32 *gpu_pair_count;
    CUDA_CHECK(cudaMalloc(&gpu_pair_count, sizeof(U32)));
    CUDA_CHECK(cudaMemset(gpu_pair_count, 0, sizeof(U32)));

    for(I32 z = 0; z < grid.dim.z; z += 1){
      for(I32 y = 0; y < grid.dim.y; y += 1){
        for(I32 x = 0; x < grid.dim.x; x += 1){
          Points main_points = grid_get_cell_points(&grid, x, y, z);

          if(main_points.count >= 2){
            U32 block_count = (main_points.count * (main_points.count - 1) / 2 + CUDA_THREAD_COUNT - 1) / CUDA_THREAD_COUNT;
            KERNEL_count_internal_pairs<<<block_count, CUDA_THREAD_COUNT>>>(
              gpu_pair_count, main_points.x, main_points.y, main_points.z, main_points.count
              );
            CUDA_CHECK(cudaGetLastError());
          }

          // TODO OPTIMIZATION(hanna): We could coalesce these to 5 regions with proper boundary handling
          // ^^^ NOTE(hanna - 2022-12-03): Didn't have time to do this :-(
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x - 1, y, z);

          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 1, y - 1, z);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 0, y - 1, z);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x - 1, y - 1, z);

          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 1, y + 1, z - 1);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 0, y + 1, z - 1);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x - 1, y + 1, z - 1);

          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 1, y + 0, z - 1);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 0, y + 0, z - 1);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x - 1, y + 0, z - 1);

          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 1, y - 1, z - 1);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x + 0, y - 1, z - 1);
          grid_cell_vs_neighbour(&grid, gpu_pair_count, main_points, x - 1, y - 1, z - 1);
        }
      }
    }

    PROFILE_begin("wait for the GPU to do its magic");
    CUDA_CHECK(cudaMemcpy(&result, gpu_pair_count, sizeof(U32), cudaMemcpyDeviceToHost));
    PROFILE_end();
    CUDA_CHECK(cudaFree(gpu_pair_count));
  }
#endif
  PROFILE_end();

  // =======
  // CLEANUP

  PROFILE_begin("cleanup");
  CUDA_CHECK(cudaFree(grid.gpu_cell_counts));
  CUDA_CHECK(cudaFree(grid.gpu_cell_point_indices));

  allocator_destroy(&allocator);
  PROFILE_end();

  PROFILE_end();
  return result;
}

//
// API
//

extern "C" GPU_API GPU_get_api(){
  GPU_API result = {
    .init = GPU_init,
    .upload_points = GPU_upload_points,
    .count_internal_pairs_async = GPU_count_internal_pairs_async,
    .count_pairs_between_async = GPU_count_pairs_between_async,
    .get_pair_count = GPU_get_pair_count,
    .grid_approach = GPU_grid_approach,
  };
  return result;
}
