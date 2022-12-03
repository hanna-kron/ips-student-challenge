/*
main.c - Solution to IPS particle simulation challenge
Author: hanna
Created: 2022-11-16

The task is to find number of pairs of points within some distance from eachother in a set 3D points.
*/

#include "shared/hanna_util.h"
#include "common.h"

#if CUDA_ENABLED
static GPUContext global_gpu_context;
#endif

//
// THREAD POOL / JOB SYSTEM / WHATEVER
// Just a simple job system I implemented for this project because I didn't have
// anything good lying around solving this problem.
//

typedef struct ThreadPool ThreadPool;

// NOTE that we only ever read from this struct in the workers so no need to make each of these
// have their own cache line.
typedef struct Worker Worker;
struct Worker{
  ThreadPool *pool;
  OSThread thread;
  U32 thread_index;
};

// NOTE(hanna): This should be set to one less than the number of processors as the main thread also participates in doing jobs.
// Letting the main thread help compared to only doing jobs in worker threads saved ~100us of time so that seems like the way to go.
#define THREAD_POOL_N_WORKERS 15

typedef struct Job Job;
struct Job{
  // These the user is supposed to set:
  void (*do_job_proc)(ThreadPool *pool, Job *job, PCG_State *prng, Allocator *allocator);
#define JOB_DATA_CAPACITY (64 - 8 - 8)
  u8 userdata[JOB_DATA_CAPACITY];

  // These fields are internal:
  Job *next;
};
CT_ASSERT(sizeof(Job) == 64); // A cache line

struct ThreadPool{
  Worker workers[THREAD_POOL_N_WORKERS];

  SpallProfile *spall;

  Job *job_buffer;
  AtomicU32 job_buffer_cursor;

  Job *first_job;

  PCG_State completion_prng;
  Allocator *completion_allocator;

  AtomicU32 quitting;

  AtomicU32 active_job_count;
  Semaphore job_semaphore;
  OSMutex mutex;
};

static ThreadPool* thread_pool_make(Allocator *allocator, SpallProfile *spall);
static Job* thread_pool_alloc_job(ThreadPool *pool);
static void thread_pool_submit_job(ThreadPool *pool, Job *job);
static void thread_pool_wait_for_completion(ThreadPool *pool);

static void _thread_pool_do_one_job_without_semaphore(ThreadPool *pool, PCG_State *prng, Allocator *allocator){
  mutex_lock(&pool->mutex);
  Job *job = pool->first_job;
  assert(job);
  pool->first_job = job->next;
  mutex_unlock(&pool->mutex);

  job->do_job_proc(pool, job, prng, allocator);

  atomic_sub_u32(&pool->active_job_count, 1);
}

static void worker_entry_point_proc(void *userdata){
  Worker *worker = (Worker*)userdata;
  ThreadPool *pool = worker->pool;

  thread_context_init(pool->spall);

  PCG_State prng = pcg_create_with_os_entropy();
  Allocator *allocator = heap_allocator_make(NULL);

  while(true){
    semaphore_wait(&pool->job_semaphore);

    if(atomic_read_u32(&pool->quitting)){
      break;
    }

    _thread_pool_do_one_job_without_semaphore(pool, &prng, allocator);
  }

  thread_context_destroy();
}

static ThreadPool* thread_pool_make(Allocator *allocator, SpallProfile *spall){
  ThreadPool *result = allocator_push_item_clear(allocator, ThreadPool);

  result->spall = spall;

  semaphore_init(&result->job_semaphore);
  mutex_init(&result->mutex);

  result->job_buffer = os_alloc_pages_nocommit(GIGABYTES(4));
  if(!result->job_buffer){
    panic("Unable to allocate job memory");
  }

  fiz(THREAD_POOL_N_WORKERS){
    Worker *worker = &result->workers[i];
    worker->pool = result;
    worker->thread_index = i;
    String name = allocator_push_printf(allocator, "Worker %I64d", i);
    worker->thread = os_start_thread(worker_entry_point_proc, worker, name);
  }

  result->completion_allocator = heap_allocator_make(NULL);
  result->completion_prng = pcg_create_with_os_entropy();

  return result;
}

static void thread_pool_destroy(ThreadPool *pool){
  atomic_store_u32(&pool->quitting, 1);

  fiz(THREAD_POOL_N_WORKERS){ // Wake up all threads
    semaphore_post(&pool->job_semaphore);
  }

  fiz(THREAD_POOL_N_WORKERS){
    Worker *worker = &pool->workers[i];
    os_join_thread(worker->thread);
  }

  allocator_destroy(&pool->completion_allocator);
}

static Job* thread_pool_alloc_job(ThreadPool *pool){
  // NOTE(hanna): Yes, this never frees or reuses memory. A proper thread pool would of course do this.
  Job *result = pool->job_buffer + atomic_add_u32(&pool->job_buffer_cursor, 1);
  return result;
}

static void thread_pool_submit_job(ThreadPool *pool, Job *job){
  atomic_add_u32(&pool->active_job_count, 1);

  mutex_lock(&pool->mutex);
  // NOTE(hanna): Setting the new job as the first element of the queue
  // can be motivated by newer jobs being hotter in the cache.
  job->next = pool->first_job;
  pool->first_job = job;
  mutex_unlock(&pool->mutex);

  semaphore_post(&pool->job_semaphore);
}

static void thread_pool_wait_for_completion(ThreadPool *pool){
  PROFILE_begin("thread_pool_wait_for_completion");
  // NOTE(hanna): It turns out that using a semaphore here rather than spinning
  // is way less efficient (200us on my laptop), hence we spin on `active_job_count`.
  while(true){
    if(atomic_read_u32(&pool->active_job_count) == 0){
      break;
    }
    if(semaphore_trywait(&pool->job_semaphore)){
      _thread_pool_do_one_job_without_semaphore(pool, &pool->completion_prng, pool->completion_allocator);
    }
  }
  PROFILE_end();
}

// ==================
//  THE ACTUAL PROGRAM
// ===================
//

//
// POINTS
//

static Rect3 v3_array_bounds(V3 *ps, U32 n_ps){
  V3 min = { F32_INFINITY, F32_INFINITY, F32_INFINITY };
  V3 max = { -F32_INFINITY, -F32_INFINITY, -F32_INFINITY };
  fjz(n_ps){
    fiz(3) min.e[i] = MINIMUM(min.e[i], ps[j].e[i]);
    fiz(3) max.e[i] = MAXIMUM(max.e[i], ps[j].e[i]);
  }
  return (Rect3){ .min = min, .max = max };
}

static V3 points_v3_at(Points points, I64 index){
  assert(0 <= index);
  assert(index < points.count);
  return (V3){ points.x[index], points.y[index], points.z[index] };
}

static void points_set_v3(Points points, U32 index, V3 p){
  assert(index < points.count);
  points.x[index] = p.x;
  points.y[index] = p.y;
  points.z[index] = p.z;
}

static Points points_allocate(Allocator *allocator, U32 count){
  Points result = {0};

  // NOTE(hanna): Here we make sure they are cache-line aligned so that the multithreaded
  // building of the KD tree data structure doesn't have threads fighting over shared cache
  // lines.
  result.count = count;
  result.x = allocator_push_memory(allocator, sizeof(F32), count, 64, false);
  result.y = allocator_push_memory(allocator, sizeof(F32), count, 64, false);
  result.z = allocator_push_memory(allocator, sizeof(F32), count, 64, false);

  return result;
}

DECLARE_ARRAY_TYPE(Points);

static Points points_copy(Points points, Allocator *allocator){
  Points result = points_allocate(allocator, points.count);

  memcpy(result.x, points.x, sizeof(F32) * points.count);
  memcpy(result.y, points.y, sizeof(F32) * points.count);
  memcpy(result.z, points.z, sizeof(F32) * points.count);

  return result;
}

static Rect3 points_bounds(Points points){
  V3 min = { F32_INFINITY, F32_INFINITY, F32_INFINITY };
  V3 max = { -F32_INFINITY, -F32_INFINITY, -F32_INFINITY };
  fjz(points.count){
    fiz(3) min.e[i] = MINIMUM(min.e[i], points.e[i][j]);
    fiz(3) max.e[i] = MAXIMUM(max.e[i], points.e[i][j]);
  }
  return (Rect3){ .min = min, .max = max };
}

static Points v3_array_to_points(Allocator *allocator, Array(V3) ps){
  Points result = points_allocate(allocator, ps.count);

  fiz(ps.count){
    result.x[i] = ps.e[i].x;
    result.y[i] = ps.e[i].y;
    result.z[i] = ps.e[i].z;
  }

  return result;
}

// TODO: AVX512 when I find compatible hardware!
// ^^^ NOTE(hanna - 2022-12-03): Didn't have time to do this :-(
static U32 AVX_count_pairs(V3 p, F32 *xs, F32 *ys, F32 *zs, U32 count){
  // PROFILE_begin("AVX_count_pairs");
  U32 result = 0;

  __m256 test_x_8xf32 = _mm256_set1_ps(p.x);
  __m256 test_y_8xf32 = _mm256_set1_ps(p.y);
  __m256 test_z_8xf32 = _mm256_set1_ps(p.z);

  __m256 critical_dist_sq_8xf32 = _mm256_set1_ps(SQUARE(CRITICAL_DISTANCE));

  I64 index = 0;
  for(; index + 8 <= count; index += 8){
    __m256 x_8xf32 = _mm256_loadu_ps((F32*)&xs[index]);
    __m256 y_8xf32 = _mm256_loadu_ps((F32*)&ys[index]);
    __m256 z_8xf32 = _mm256_loadu_ps((F32*)&zs[index]);

    x_8xf32 = _mm256_sub_ps(x_8xf32, test_x_8xf32);
    x_8xf32 = _mm256_mul_ps(x_8xf32, x_8xf32);

    y_8xf32 = _mm256_sub_ps(y_8xf32, test_y_8xf32);
    y_8xf32 = _mm256_mul_ps(y_8xf32, y_8xf32);

    z_8xf32 = _mm256_sub_ps(z_8xf32, test_z_8xf32);
    z_8xf32 = _mm256_mul_ps(z_8xf32, z_8xf32);

    __m256 dist_sq_8xf32 = _mm256_add_ps(x_8xf32, _mm256_add_ps(y_8xf32, z_8xf32));

    int mask = _mm256_movemask_ps(_mm256_cmp_ps(dist_sq_8xf32, critical_dist_sq_8xf32, _CMP_LE_OS));
    result += __builtin_popcount(mask);
  }
  for(; index + 1 <= count; index += 1){
    if(v3_distance_squared(p, (V3){ xs[index], ys[index], zs[index] }) <= SQUARE(CRITICAL_DISTANCE)){
      result += 1;
    }
  }

  // PROFILE_end();
  return result;
}

static U32 AVX_count_pairs_between(F32 *x0, F32 *y0, F32 *z0, U32 count0, F32 *x1, F32 *y1, F32 *z1, U32 count1){
  U32 result = 0;

  fiz(count0){
    V3 p = { x0[i], y0[i], z0[i] };
    result += AVX_count_pairs(p, x1, y1, z1, count1);
  }

  return result;
}

static U32 SSE_count_pairs(V3 p, F32 *xs, F32 *ys, F32 *zs, U32 count){
  U32 result = 0;

  __m128 test_x_4xf32 = _mm_set1_ps(p.x);
  __m128 test_y_4xf32 = _mm_set1_ps(p.y);
  __m128 test_z_4xf32 = _mm_set1_ps(p.z);

  __m128 critical_dist_sq_4xf32 = _mm_set1_ps(SQUARE(CRITICAL_DISTANCE));

  I64 index = 0;
  for(; index + 4 <= count; index += 4){
    __m128 x_4xf32 = _mm_loadu_ps((F32*)&xs[index]);
    __m128 y_4xf32 = _mm_loadu_ps((F32*)&ys[index]);
    __m128 z_4xf32 = _mm_loadu_ps((F32*)&zs[index]);

    x_4xf32 = _mm_sub_ps(x_4xf32, test_x_4xf32);
    x_4xf32 = _mm_mul_ps(x_4xf32, x_4xf32);

    y_4xf32 = _mm_sub_ps(y_4xf32, test_y_4xf32);
    y_4xf32 = _mm_mul_ps(y_4xf32, y_4xf32);

    z_4xf32 = _mm_sub_ps(z_4xf32, test_z_4xf32);
    z_4xf32 = _mm_mul_ps(z_4xf32, z_4xf32);

    __m128 dist_sq_4xf32 = _mm_add_ps(x_4xf32, _mm_add_ps(y_4xf32, z_4xf32));

    int mask = _mm_movemask_ps(_mm_cmple_ps(dist_sq_4xf32, critical_dist_sq_4xf32));
#if 1 // NOTE(hanna): It looks like the table variant is slightly faster than popcount
    int table[16] = {
      0, /* 0000 */ 1, /* 0001 */ 1, /* 0010 */ 2, /* 0011 */
      1, /* 0100 */ 2, /* 0101 */ 2, /* 0110 */ 3, /* 0111 */
      1, /* 1000 */ 2, /* 1001 */ 2, /* 1010 */ 3, /* 1011 */
      2, /* 1100 */ 3, /* 1101 */ 3, /* 1110 */ 4, /* 1111 */
    };
    result += table[mask];
#else
    result += __builtin_popcount(mask);
#endif
  }
  for(; index + 1 <= count; index += 1){
    if(v3_distance_squared(p, (V3){ xs[index], ys[index], zs[index] }) <= SQUARE(CRITICAL_DISTANCE)){
      result += 1;
    }
  }

  return result;
}

//
// QUICKSELECT
//

// Partitions `ps` into elements less than k:th smallest element and greater than that element
// and returns the index of the this k:th smallest element.
static I64 quickselect(Points ps, PCG_State *prng, I64 left, I64 right, I64 k, int axis){
  I64 result = 0;

  assert(right >= left);

#define SWAP(_a_, _b_) \
  { \
    f32_swap(&ps.x[(_a_)], &ps.x[(_b_)]); \
    f32_swap(&ps.y[(_a_)], &ps.y[(_b_)]); \
    f32_swap(&ps.z[(_a_)], &ps.z[(_b_)]); \
  }

  while(true){
    if(left == right){
      result = left;
      break;
    }else{
      #if 0
      I64 pivot_index = right;
      #else // This version seems to perform slightly better
      I64 pivot_index = left + pcg_random_u32(prng) % (right - left);
      #endif

      { // Partition
        F32 pivot_value = ps.e[axis][pivot_index];
        SWAP(pivot_index, right);
        pivot_index = left;

        I64 index = left;
        #if 0
        for(; index + 4 <= right; index += 4){
          __m128 v_4xf32 = _mm_loadu_ps(ps.e[axis] + index);
          __m128 pivot_value_4xf32 = _mm_set1_ps(pivot_value);

          int mask = _mm_movemask_ps(_mm_cmple_ps(v_4xf32, pivot_value_4xf32));

          if(mask == 0){
            // need to do nothing!
          }else{
            #if 0
            fiz(4){
              if(((mask >> i) & 1)){
                SWAP(index + i, pivot_index);
                pivot_index += 1;
              }
            }
            #else // Unrolled version is noticably faster
            if((mask & 1)){ SWAP(index + 0, pivot_index); pivot_index += 1; }
            if((mask & 2)){ SWAP(index + 1, pivot_index); pivot_index += 1; }
            if((mask & 4)){ SWAP(index + 2, pivot_index); pivot_index += 1; }
            if((mask & 8)){ SWAP(index + 3, pivot_index); pivot_index += 1; }
            #endif
          }
        }
        #else
        __m256 pivot_value_8xf32 = _mm256_set1_ps(pivot_value);
        for(; index + 8 <= right; index += 8){
          __m256 v_8xf32 = _mm256_loadu_ps(ps.e[axis] + index);

          int mask = _mm256_movemask_ps(_mm256_cmp_ps(v_8xf32, pivot_value_8xf32, _CMP_LT_OS));

          // TODO OPTIMIZATION: Here we could do the swapping also with SIMD.
          // Perhaps we should try using AVX512 compress/expand stuff when I find
          // some AVX512 hardware.
          // ^^^ NOTE(hanna - 2022-12-03): Didn't have time to do this :-(

          // Explicitly unrolled is noticably faster than a loop
          if((mask &  0x1)){ SWAP(index + 0, pivot_index); pivot_index += 1; }
          if((mask &  0x2)){ SWAP(index + 1, pivot_index); pivot_index += 1; }
          if((mask &  0x4)){ SWAP(index + 2, pivot_index); pivot_index += 1; }
          if((mask &  0x8)){ SWAP(index + 3, pivot_index); pivot_index += 1; }
          if((mask & 0x10)){ SWAP(index + 4, pivot_index); pivot_index += 1; }
          if((mask & 0x20)){ SWAP(index + 5, pivot_index); pivot_index += 1; }
          if((mask & 0x40)){ SWAP(index + 6, pivot_index); pivot_index += 1; }
          if((mask & 0x80)){ SWAP(index + 7, pivot_index); pivot_index += 1; }
        }
        #endif

        for(; index + 1 <= right; index += 1){
          if(ps.e[axis][index] < pivot_value){
            SWAP(index, pivot_index);
            pivot_index += 1;
          }
        }
        SWAP(right, pivot_index);
      }

      if(k == pivot_index){
        result = pivot_index;
        break;
      }else if(k < pivot_index){
        left = left;
        right = pivot_index - 1;
      }else{
        left = pivot_index + 1;
        right = right;
      }
    }
  }

  #undef SWAP

  return result;
}

//
// KD-TREE
//
// A very helpful video series: https://www.youtube.com/watch?v=-CnrkvpHZAY
//
// The time complexity of building the tree is O(n log n) where n is the number of points.
// The time complexity of querying the tree for each of the n points is O(n * (n^(2/3) + m)) where
// m depends on the density of points (proportional to the number of pairs per volume of space of something
// like that). This is apperently a high overestimation though, so ...
//
// ... TODO: Lets measure the time complexity for constant m, i.e. constant point density.
// ^^^ NOTE(hanna - 2022-12-03): Didn't have time to do this :-(
//
// The reason why we don't care so badly about making KDNode be a very small struct
// is because for the problem size we deal with, 16k points, we don't get all that
// many nodes, so they should all be able to stay in cache anyway (remember we store
// ~512 points per leaf node!).
//
//

typedef struct KDNode KDNode, *KDNodePtr;
DECLARE_ARRAY_TYPE(KDNodePtr);
__attribute__((__packed__)) struct
KDNode{
  // ============================
  // SET DIRECTLY WHEN ALLOCATED:
  KDNode *parent;
  U32 ps_begin;
  U32 ps_end;
  Rect3 bounds;
  int depth;

  // ===============
  // AFTER BUILDING:
  bool is_leaf;

  // These are for leaf nodes:
  U32 leaf_index; // Index in flattened array of leaf nodes.

  // These are for internal nodes:
  KDNode *left, *right;
  F32 split_value; // position of the splitting line
};

typedef struct KDTree KDTree;
struct KDTree{
  KDNode *root;
  Points points;

  // After building the tree:
  #if CUDA_ENABLED
  Points gpu_points;
  #endif
  Array(KDNodePtr) leaf_nodes; // Flattened array of all the leaf nodes
};

static void kd_tree_flatten(KDTree *tree, KDNode *node){
  assert(node != NULL);
  if(node->is_leaf){
    node->leaf_index = tree->leaf_nodes.count;
    array_push(&tree->leaf_nodes, node);
  }else{
    kd_tree_flatten(tree, node->left);
    kd_tree_flatten(tree, node->right);
  }
}

static KDNode *kd_node_make(Allocator *allocator, KDNode *parent, U32 ps_begin, U32 ps_end, Rect3 bounds, int depth){
  KDNode *result = allocator_push_item_clear(allocator, KDNode);

  result->parent = parent;
  result->ps_begin = ps_begin;
  result->ps_end = ps_end;
  result->bounds = bounds;
  result->depth = depth;

  return result;
}

static void kd_build_one_node(KDTree *tree, KDNode *node, PCG_State *prng, Allocator *allocator){
  PROFILE_begin("kd_build_one_node");

  assert(node->ps_begin <= node->ps_end);
  assert(node->ps_end <= tree->points.count);

  int axis = node->depth % 3;

  // NOTE(hanna): The constant 512 here has been fine-tuned for maximal performance.
  // Due to the internals of the GPU code this cannot be too big.
  if(node->ps_end - node->ps_begin <= 512){
    node->is_leaf = true;
  }else{
    u64 wanted_rank = node->ps_begin + (node->ps_end - node->ps_begin) / 2;
    wanted_rank &= ~LIT_U64(15); // Align to a cache line boundary
    U32 median_index = quickselect(tree->points, prng, node->ps_begin, node->ps_end - 1, wanted_rank, axis);
    node->split_value = tree->points.e[axis][median_index];

    Rect3 left_bounds = node->bounds;
    Rect3 right_bounds = node->bounds;

    left_bounds.max.e[axis] = node->split_value;
    right_bounds.min.e[axis] = node->split_value;

    node->left = kd_node_make(allocator, node, node->ps_begin, median_index, left_bounds, node->depth + 1);
    node->right = kd_node_make(allocator, node, median_index, node->ps_end, right_bounds, node->depth + 1);
  }

  PROFILE_end();
}

static void kd_build_node_recursive(KDTree *tree, KDNode *node, PCG_State *prng, Allocator *allocator){
  kd_build_one_node(tree, node, prng, allocator);

  if(!node->is_leaf){
    kd_build_node_recursive(tree, node->left, prng, allocator);
    kd_build_node_recursive(tree, node->right, prng, allocator);
  }
}

// !! NOTE THAT `ps` WILL BE MODIFIED WHILE BUILDING THE TREE !!
static KDTree kd_tree_create(Points ps, Allocator *allocator){
  KDTree result = {0};
  result.points = ps;

  // NOTE(hanna - 2022-12-01): Computing the exact bounds takes time and having a notion of the root and its immidiate descendants being "infinite"
  // makes some of the code cleaner and more also efficient.
  Rect3 bounds = { vec3_set1(-F32_INFINITY), vec3_set1(F32_INFINITY) };

  result.root = kd_node_make(allocator, NULL, 0, ps.count, bounds, 0);
  return result;
}

static U32 kd_count(KDTree *tree, KDNode *node, V3 p, I64 point_index, Rect3 search_bounds){
  U32 result = 0;

  U32 axis = node->depth % 3;

  if(point_index < node->ps_end){
    if(node->is_leaf){
      if(point_index < node->ps_begin){
        Points points = points_subarray(tree->points, node->ps_begin, node->ps_end);
        result += AVX_count_pairs(p, points.x, points.y, points.z, points.count);
      }
    }else{
      if(search_bounds.min.e[axis] <= node->split_value){
        Rect3 bounds = search_bounds;
        // bounds.max.e[axis] = MINIMUM(bounds.max.e[axis], node->split_value);
        result += kd_count(tree, node->left, p, point_index, bounds);
      }
      if(search_bounds.max.e[axis] >= node->split_value){
        Rect3 bounds = search_bounds;
        // bounds.min.e[axis] = MAXIMUM(bounds.min.e[axis], node->split_value);
        result += kd_count(tree, node->right, p, point_index, bounds);
      }
    }
  }

  return result;
}

static U32 kd_node_find_all_pairs(KDTree *tree, KDNode *node){
  PROFILE_begin("kd_node_find_all_pairs");

  U32 result = 0;

  assert(node->is_leaf);

  assert(node->ps_begin <= node->ps_end);
  assert(node->ps_end <= tree->points.count);
  for(I64 i = node->ps_begin; i < node->ps_end; i += 1){
    V3 p = points_v3_at(tree->points, i);

#if !CUDA_ENABLED || !KD_PARALLEL_COUNTING
    // Count internal pairs.
    result += AVX_count_pairs(p,
                              tree->points.x + i + 1,
                              tree->points.y + i + 1,
                              tree->points.z + i + 1,
                              node->ps_end - i - 1);
#endif

    Rect3 search_bounds = rect3_from_center_and_extents(p, vec3_set1(CRITICAL_DISTANCE));
    result += kd_count(tree, tree->root, p, i, search_bounds);
  }

  PROFILE_end();
  return result;
}

//~ Query

typedef struct KDTest KDTest;
struct KDTest{
  U32 leaf_index;
  V3 p;
};
typedef struct KDTests KDTests;
struct KDTests{
  KDTest *buffer; // Big chunk of virtual memory
  U32 cursor;
  U32 capacity;
};

static void kd_query(KDTree *tree, KDNode *node, KDTests *tests, V3 p, I64 point_index, Rect3 search_bounds){
  int axis = node->depth % 3;

  if(point_index < node->ps_end){
    if(node->is_leaf){
      if(point_index < node->ps_begin){
        if(tests->cursor >= tests->capacity){
          panic("Ran out of space for tests!");
        }
        tests->buffer[tests->cursor++] = (KDTest){ .leaf_index = node->leaf_index, .p = p };
      }
    }else{
      if(search_bounds.min.e[axis] <= node->split_value){
        Rect3 bounds = search_bounds;
        // bounds.max.e[axis] = MINIMUM(bounds.max.e[axis], node->split_value);
        kd_query(tree, node->left, tests, p, point_index, bounds);
      }
      if(search_bounds.max.e[axis] >= node->split_value){
        Rect3 bounds = search_bounds;
        // bounds.min.e[axis] = MAXIMUM(bounds.min.e[node->axis], node->split_value);
        kd_query(tree, node->right, tests, p, point_index, bounds);
      }
    }
  }
}

static void kd_query_bottom_up(KDTree *tree, KDNode *node, KDTests *tests, V3 p, I64 point_index, Rect3 search_bounds){
  while(true){
    assert(node);
    if(rect3_contains(node->bounds, search_bounds)){
      kd_query(tree, node, tests, p, point_index, search_bounds);
      break;
    }
    node = node->parent;
  }
}

// Query for all tests which are between nodes.
static void kd_node_query_all_external_tests(KDTree *tree, KDNode *node, KDTests *tests){
  PROFILE_begin("kd_node_query_all_external_tests");

  assert(node->is_leaf);

  assert(node->ps_begin <= node->ps_end);
  assert(node->ps_end <= tree->points.count);
  for(I64 j = node->ps_begin; j < node->ps_end; j += 1){
    V3 p = points_v3_at(tree->points, j);

    Rect3 search_bounds = rect3_from_center_and_extents(p, vec3_set1(CRITICAL_DISTANCE));
    kd_query_bottom_up(tree, node, tests, p, j, search_bounds);
  }

  PROFILE_end();
}

//
// 3D RANGE TREE
// Thanks to Erik Demaine and MIT OpenCourseWare for this material: https://www.youtube.com/watch?v=xVka6z1hu-I
// Thanks to Philipp Kindermann for this very helpful material: https://www.youtube.com/watch?v=kRW0HIhm0zc
//

typedef struct RNode RNode;
struct RNode{
  bool is_leaf;
  U8 axis;

  // Leaf:
  Points points;

  // Internal node:
  F32 maximum;
  RNode *augmentation;
  RNode *left, *right;
};

typedef struct RTree RTree;
struct RTree{
  RNode *root;
};

static RNode* rtree_make_node(Allocator *allocator, U8 axis){
  RNode *result = allocator_push_item_clear(allocator, RNode);
  assert(axis < 3);
  result->axis = axis;
  return result;
}

static RTree rtree_create(Allocator *allocator){
  RTree result = {0};
  result.root = rtree_make_node(allocator, 0);
  return result;
}

// NOTE: This routine modifies and references the memory of `ps`
static void rtree_node_build(RTree *tree, RNode *node, PCG_State *prng, Allocator *allocator, Points ps){
  if(ps.count <= 256){
    node->is_leaf = true;
    node->maximum = -F32_INFINITY;
    fiz(ps.count){
      node->maximum = MAXIMUM(node->maximum, ps.e[node->axis][i]);
    }
    node->points = ps;
  }else{
    // BUILD AUGMENTATION
    if(node->axis < 2){
      node->augmentation = rtree_make_node(allocator, node->axis + 1);
      Points copied_points = points_copy(ps, allocator);
      rtree_node_build(tree, node->augmentation, prng, allocator, copied_points);
    }

    // PARTITION AS PREPARATION FOR BUILDING CHILDREN
    I64 median_index = ps.count / 2;
    quickselect(ps, prng, 0, ps.count - 1, median_index, node->axis);

    // RECURSE TO CHILDREN
    node->left = rtree_make_node(allocator, node->axis);
    rtree_node_build(tree, node->left, prng, allocator, points_subarray(ps, 0, median_index));

    node->right = rtree_make_node(allocator, node->axis);
    rtree_node_build(tree, node->right, prng, allocator, points_subarray(ps, median_index, ps.count));

    // UPDATE PARENT
    node->maximum = MAXIMUM(node->left->maximum, node->right->maximum);
  }
}

static U32 rtree_count_include_all(RTree *tree, RNode *node, V3 p){
  U32 result = 0;
  if(node->is_leaf){
    result += AVX_count_pairs(p, node->points.x, node->points.y, node->points.z, node->points.count);
  }else{
    result += rtree_count_include_all(tree, node->left, p);
    result += rtree_count_include_all(tree, node->right, p);
  }
  return result;
}

static U32 rtree_count(RTree *tree, RNode *node, V3 p, Rect3 search_bounds){
  U32 result = 0;

  if(node->is_leaf){
    result += AVX_count_pairs(p, node->points.x, node->points.y, node->points.z, node->points.count);
  }else{
    bool A = (search_bounds.min.e[node->axis] < node->left->maximum);
    bool B = (search_bounds.max.e[node->axis] < node->left->maximum);

    if(A && !B){
      if(node->augmentation){
        result += rtree_count(tree, node->augmentation, p, search_bounds);
      }else{
        result += rtree_count_include_all(tree, node->right, p);
        result += rtree_count_include_all(tree, node->left, p);
      }
    }else if(A && B){
      result += rtree_count(tree, node->left, p, search_bounds);
    }else if(!A && !B){
      result += rtree_count(tree, node->right, p, search_bounds);
    }else{
      panic("GAAAAAAAAH");
    }
  }

  return result;
}

//
// KD Tree Parallelization
//

#define KD_PARALLEL_BUILDING 1
#define KD_PARALLEL_COUNTING 1

typedef struct KDTreeContext KDTreeContext;
struct KDTreeContext{
  // For (possibly parallel) building:
  KDTree tree;

  // For parallel pair counting, stage 1:
  AtomicU32 node_index;

  // For parallel pair counting, stage 2:
  AtomicU32 pair_count;
};

#if KD_PARALLEL_BUILDING
typedef struct KDBuildNodeJobData KDBuildNodeJobData;
struct KDBuildNodeJobData{
  KDTreeContext *kd;
  KDNode *node;
};
CT_ASSERT(sizeof(KDBuildNodeJobData) <= JOB_DATA_CAPACITY);

static void kd_tree_job_build_proc(ThreadPool *pool, Job *job, PCG_State *prng, Allocator *allocator);

static void kd_tree_submit_build_job(ThreadPool *pool, KDTreeContext *kd, KDNode *node){
  assert(node);
  Job *job = thread_pool_alloc_job(pool);
  job->do_job_proc = kd_tree_job_build_proc;
  KDBuildNodeJobData *data = (KDBuildNodeJobData*)job->userdata;
  data->kd = kd;
  data->node = node;
  thread_pool_submit_job(pool, job);
}

static void kd_tree_job_build_proc(ThreadPool *pool, Job *job, PCG_State *prng, Allocator *allocator){
  KDBuildNodeJobData *data = (KDBuildNodeJobData*)job->userdata;
  KDTreeContext *kd = data->kd;

  // NOTE(hanna): This is a loop instead of submitting two jobs as that increased performance by 20us on my desktop (4 threads).
  // Not a huge gain but at least its something.
  KDNode *node = data->node;
  while(true){
    kd_build_one_node(&kd->tree, node, prng, allocator);

    if(!node->is_leaf){
      kd_tree_submit_build_job(pool, kd, node->right);
      node = node->left;
    }else{
      break;
    }
  }
}
#endif

typedef struct CalculateResult CalculateResult;
struct CalculateResult{
  U32 pair_count;
  U64 building_us;
  U64 counting_us;
};

//
// !! COUNT VARIANT 1 !!
//

#if KD_PARALLEL_COUNTING

typedef struct KDQueryNodeJobData KDQueryNodeJobData;
struct KDQueryNodeJobData{
  KDTreeContext *kd;
  KDTests tests;
};
CT_ASSERT(sizeof(KDQueryNodeJobData) <= JOB_DATA_CAPACITY);

static void kd_tree_job_query_proc(ThreadPool *pool, Job *job, PCG_State *prng, Allocator *allocator){
  KDQueryNodeJobData *data = (KDQueryNodeJobData*)job->userdata;
  KDTreeContext *kd = data->kd;

  while(true){
    U32 index = atomic_add_u32(&kd->node_index, 1);
    if(index >= kd->tree.leaf_nodes.count) break;

    KDNode *node = kd->tree.leaf_nodes.e[index];
    assert(node->is_leaf);

    kd_node_query_all_external_tests(&kd->tree, node, &data->tests);
  }
}

typedef struct KDCountExternalJobData KDCountExternalJobData;
struct KDCountExternalJobData{
  KDTreeContext *kd;
  U32 index;
  Points test_points;
};
CT_ASSERT(sizeof(KDCountExternalJobData) <= JOB_DATA_CAPACITY);

static void kd_tree_job_count_external_proc(ThreadPool *pool, Job *job, PCG_State *prng, Allocator *allocator){
  PROFILE_begin("kd_tree_job_count_external_proc");

  KDCountExternalJobData *data = (KDCountExternalJobData*)job->userdata;
  KDTreeContext *kd = data->kd;

  KDNode *node = kd->tree.leaf_nodes.e[data->index];

  Points node_points = points_subarray(kd->tree.points, node->ps_begin, node->ps_end);
  Points test_points = data->test_points;

  U32 pair_count = 0;
#if !CUDA_ENABLED
  for(I64 i = node->ps_begin; i < node->ps_end; i += 1){
    V3 p = points_v3_at(kd->tree.points, i);
    pair_count += AVX_count_pairs(p,
                                  kd->tree.points.x + i + 1,
                                  kd->tree.points.y + i + 1,
                                  kd->tree.points.z + i + 1,
                                  node->ps_end - i - 1);
  }
#endif

  pair_count += AVX_count_pairs_between(node_points.x, node_points.y, node_points.z, node_points.count,
                                        test_points.x, test_points.y, test_points.z, test_points.count);
  atomic_add_u32(&kd->pair_count, pair_count);

  PROFILE_end();
}
#endif

#if KD_PARALLEL_COUNTING
static void count_pairs_kd_tree_variant1(ThreadPool *pool, Allocator *allocator, KDTreeContext *kd, CalculateResult *result){
  PROFILE_begin("count_pairs_kd_tree_variant1");

  Job *query_jobs[THREAD_POOL_N_WORKERS] = {0};

  // =========================================
  // STEP 0: Submit query_jobs for querying the tree

  PROFILE_begin("step 0");

  fiz(THREAD_POOL_N_WORKERS){
    Job *job = query_jobs[i] = thread_pool_alloc_job(pool);

    job->do_job_proc = kd_tree_job_query_proc;
    KDQueryNodeJobData *data = (KDQueryNodeJobData*)job->userdata;
    data->kd = kd;

    PROFILE_begin("aquire virtual memory");
    data->tests.capacity = MEGABYTES(256) / sizeof(KDTest);
    data->tests.buffer = os_alloc_pages_nocommit(data->tests.capacity * sizeof(KDTest));
    if(!data->tests.buffer){ panic("Unable to aquire memory for KDTests buffer"); }
    PROFILE_end();

    thread_pool_submit_job(pool, job);
  }

  PROFILE_end();

  // ============================
  // STEP 1: Count internal pairs.

#if CUDA_ENABLED
  PROFILE_begin("count internal pairs async");
  fiz(kd->tree.leaf_nodes.count){
    KDNode *node = kd->tree.leaf_nodes.e[i];

    Points gpu_points = points_subarray(kd->tree.gpu_points, node->ps_begin, node->ps_end);
    global_gpu_api.count_internal_pairs_async(&global_gpu_context, gpu_points.x, gpu_points.y, gpu_points.z, gpu_points.count);
  }
  PROFILE_end();
#endif // CUDA_ENABLED

  thread_pool_wait_for_completion(pool);

  // ==================
  // PREPARE FOR STEP 2

  PROFILE_begin("prepare for step 2");
  U32 *tests_per_leaf_counts = allocator_push_items_clear(allocator, U32, kd->tree.leaf_nodes.count);
  U32 leaf_counts_total = 0;

  fjz(THREAD_POOL_N_WORKERS){
    Job *job = query_jobs[j];

    KDTests tests = ((KDQueryNodeJobData*) job->userdata)->tests;
    fiz(tests.cursor){
      KDTest test = tests.buffer[i];

      tests_per_leaf_counts[test.leaf_index] += 1;
      leaf_counts_total += 1;
    }
  }

  // NOTE(hanna): I tried allocating pinned memory for improving transfer speeds here, but it turns out to be **way** too slow
  // to allocate. It literally takes like 600us on my desktop so doing that is a no-go.
  Points external_points = points_allocate(allocator, leaf_counts_total);
  leaf_counts_total = 0;
  Points *tests_per_leaf = allocator_push_items_noclear(allocator, Points, kd->tree.leaf_nodes.count);
  fiz(kd->tree.leaf_nodes.count){
    tests_per_leaf[i] = points_subarray(external_points, leaf_counts_total, leaf_counts_total + tests_per_leaf_counts[i]);
    leaf_counts_total += tests_per_leaf_counts[i];
    tests_per_leaf_counts[i] = 0;
  }

  fjz(THREAD_POOL_N_WORKERS){
    Job *job = query_jobs[j];

    KDTests tests = ((KDQueryNodeJobData*) job->userdata)->tests;
    fiz(tests.cursor){
      KDTest test = tests.buffer[i];

      points_set_v3(tests_per_leaf[test.leaf_index], tests_per_leaf_counts[test.leaf_index], test.p);
      tests_per_leaf_counts[test.leaf_index] += 1;
    }
  }
  PROFILE_end();

#define EXTERNAL_ON_GPU 0

#if CUDA_ENABLED && EXTERNAL_ON_GPU
  PROFILE_begin("upload points for external point tests");
  Points gpu_external_points = global_gpu_api.upload_points(CUDA_STREAM_HANDLE_DEFAULT, external_points);
  PROFILE_end();

  Points *gpu_tests_per_leaf = allocator_push_items_noclear(allocator, Points, kd->tree.leaf_nodes.count);
  leaf_counts_total = 0;
  fiz(kd->tree.leaf_nodes.count){
    gpu_tests_per_leaf[i] = points_subarray(gpu_external_points, leaf_counts_total, leaf_counts_total + tests_per_leaf_counts[i]);
    leaf_counts_total += tests_per_leaf_counts[i];
    tests_per_leaf_counts[i] = 0;
  }
#endif

  // ===============================
  // STEP 2: Count "external" pairs.

  PROFILE_begin("step 2");

  #if CUDA_ENABLED && EXTERNAL_ON_GPU
  PROFILE_begin("start kernels for external points GPU");
  fiz(kd->tree.leaf_nodes.count){
    KDNode *node = kd->tree.leaf_nodes.e[i];

    Points gpu_node_points = points_subarray(kd->tree.gpu_points, node->ps_begin, node->ps_end);
    Points gpu_test_points = gpu_tests_per_leaf[i];

    global_gpu_api.count_pairs_between_async(&global_gpu_context,
                                             gpu_node_points.x, gpu_node_points.y, gpu_node_points.z, gpu_node_points.count,
                                             gpu_test_points.x, gpu_test_points.y, gpu_test_points.z, gpu_test_points.count);
  }
  PROFILE_end();
  #else
  fiz(kd->tree.leaf_nodes.count){
    Job *job = thread_pool_alloc_job(pool);

    job->do_job_proc = kd_tree_job_count_external_proc;
    KDCountExternalJobData *data = (KDCountExternalJobData*)job->userdata;
    data->kd = kd;
    data->index = i;
    data->test_points = tests_per_leaf[i];

    thread_pool_submit_job(pool, job);
  }

  thread_pool_wait_for_completion(pool);
  #endif

  result->pair_count += atomic_read_u32(&kd->pair_count);

#if CUDA_ENABLED
  PROFILE_begin("wait for GPU to do all the work");
  result->pair_count += global_gpu_api.get_pair_count(&global_gpu_context);
  PROFILE_end();
#endif // CUDA_ENABLED
  PROFILE_end();

  PROFILE_end();
}
#endif // KD_PARALLEL_COUNTING

//
// !! COUNT VARIANT 2 !!
// This is a simpler variant that counts all "external" pairs (i.e. pairs that cross node boundaries) on the CPU in parellell
// but counts all "internal" pairs (i.e. those within nodes) on the GPU in parallell with the counting on the CPU.
// As of 2022-11-26 variant 1 and variant 2 are equally fast.
//

#if KD_PARALLEL_COUNTING

typedef struct KDCountNodeJobData KDCountNodeJobData;
struct KDCountNodeJobData{
  KDTreeContext *kd;
};

static void kd_tree_job_count_proc(ThreadPool *pool, Job *job, PCG_State *prng, Allocator *allocator){
  KDCountNodeJobData *data = (KDCountNodeJobData*)job->userdata;
  KDTreeContext *kd = data->kd;

  U32 pair_count = 0;

  while(true){
    U32 index = atomic_add_u32(&kd->node_index, 1);
    if(index >= kd->tree.leaf_nodes.count) break;

    KDNode *node = kd->tree.leaf_nodes.e[index];
    assert(node->is_leaf);

    pair_count += kd_node_find_all_pairs(&kd->tree, node);
  }

  atomic_add_u32(&kd->pair_count, pair_count);
}

static void count_pairs_kd_tree_variant2(ThreadPool *pool, Allocator *allocator, KDTreeContext *kd, CalculateResult *result){
  // STEP 1: Count all pairs.

  Job *jobs[THREAD_POOL_N_WORKERS] = {0};

  PROFILE_begin("submit jobs");
  fiz(THREAD_POOL_N_WORKERS){
    Job *job = jobs[i] = thread_pool_alloc_job(pool);

    job->do_job_proc = kd_tree_job_count_proc;
    KDCountNodeJobData *data = (KDCountNodeJobData*)job->userdata;
    data->kd = kd;

    thread_pool_submit_job(pool, job);
  }
  PROFILE_end();

  #if CUDA_ENABLED
  fiz(kd->tree.leaf_nodes.count){
    KDNode *node = kd->tree.leaf_nodes.e[i];

    Points gpu_points = points_subarray(kd->tree.gpu_points, node->ps_begin, node->ps_end);
    global_gpu_api.count_internal_pairs_async(&global_gpu_context, gpu_points.x, gpu_points.y, gpu_points.z, gpu_points.count);
  }
  #endif

  thread_pool_wait_for_completion(pool);

  result->pair_count += atomic_read_u32(&kd->pair_count);

#if CUDA_ENABLED
  result->pair_count += global_gpu_api.get_pair_count(&global_gpu_context);
#endif
}
#endif // KD_PARALLEL_COUNTING


static CalculateResult calculate_pair_count_with_kd_tree(ThreadPool *pool, Points passed_points){
  PROFILE_begin("calculate_pair_count_with_kd_tree");

  PROFILE_begin("preparation");
  CalculateResult result = {0};

  Allocator *allocator = heap_allocator_make(NULL);

  #if CUDA_ENABLED
  global_gpu_api.init(&global_gpu_context, global_thread_context);
  #endif

  Points ps = points_copy(passed_points, allocator);

  PROFILE_end();

  KDTreeContext kd = {0};

  //
  // !! BUILD TREE CODE !!
  //

  PROFILE_begin("build tree");
  U64 build_tree_start_us = os_get_monotonic_time_us();

  kd.tree = kd_tree_create(ps, allocator);

#if KD_PARALLEL_BUILDING
  kd_tree_submit_build_job(pool, &kd, kd.tree.root);
  thread_pool_wait_for_completion(pool);
#else
  PCG_State prng = pcg_create_with_os_entropy();
  kd_build_node_recursive(&kd.tree, kd.tree.root, &prng, allocator);
#endif

  PROFILE_begin("flatten kd tree");
  kd.tree.leaf_nodes = array_create(KDNodePtr, allocator);
  kd_tree_flatten(&kd.tree, kd.tree.root);
  PROFILE_end();

  #if CUDA_ENABLED
  PROFILE_begin("upload points to GPU");
  kd.tree.gpu_points = global_gpu_api.upload_points(kd.tree.points);
  PROFILE_end();
  #endif
  U64 build_tree_end_us = os_get_monotonic_time_us();

  result.building_us = build_tree_end_us - build_tree_start_us;
  PROFILE_end();

  //
  // !! COUNT NODES CODE !!
  //

  PROFILE_begin("count nodes");
  U64 count_start_us = os_get_monotonic_time_us();
  #if KD_PARALLEL_COUNTING
  count_pairs_kd_tree_variant1(pool, allocator, &kd, &result);
  #else
  fiz(kd.tree.leaf_nodes.count){
    KDNode *node = kd.tree.leaf_nodes.e[i];
    assert(node->is_leaf);
    result.pair_count += kd_node_find_all_pairs(&kd.tree, node);
  }
  #endif
  U64 count_end_us = os_get_monotonic_time_us();
  result.counting_us = count_end_us - count_start_us;
  PROFILE_end();

  allocator_destroy(&allocator);

  PROFILE_end();

  return result;
}

//
// MORE RANGE TREE CODE
//

static CalculateResult calculate_pair_count_with_range_tree(ThreadPool *pool, Points points){
  CalculateResult result = {0};

  Allocator *allocator = heap_allocator_make(NULL);
  PCG_State prng = pcg_create_with_os_entropy();

  // ==============
  // !! BUILDING !!
  // ==============

  U64 build_start_us = os_get_monotonic_time_us();
  RTree tree = rtree_create(allocator);
  rtree_node_build(&tree, tree.root, &prng, allocator, points_copy(points, allocator));
  U64 build_end_us = os_get_monotonic_time_us();

  result.building_us = build_end_us - build_start_us;

  // ==============
  // !! COUNTING !!
  // ==============

  U64 count_start_us = os_get_monotonic_time_us();
  fiz(points.count){
    V3 p = points_v3_at(points, i);
    Rect3 bounds = rect3_from_center_and_extents(p, vec3_set1(CRITICAL_DISTANCE));
    result.pair_count += rtree_count(&tree, tree.root, p, bounds) - 1;
  }
  U64 count_end_us = os_get_monotonic_time_us();

  assert((result.pair_count & 1) == 0);
  result.pair_count /= 2;

  result.counting_us = count_end_us - count_start_us;

  //====================

  allocator_destroy(&allocator);

  return result;
}

//
// CALCULATE WITH GPU GRID APPROACH
//

#if CUDA_ENABLED
static CalculateResult calculate_pair_count_with_gpu_grid(Points points){
  Rect3 bounds = points_bounds(points);

  CalculateResult result = {0};

  Points gpu_points = global_gpu_api.upload_points(points);
  result.pair_count = global_gpu_api.grid_approach(gpu_points, bounds);

  return result;
}
#endif // CUDA_ENABLED

//
// MAIN
//

static Points parse_points(Allocator *allocator, String path){
  EntireFile content = read_entire_file(path, allocator);
  if(!content.ok) panic("unable to read positions.xyz :-(");

  String source = { .data = content.content, .size = content.size };

  I64 point_count = 0;
  for(I64 cursor = 0; cursor < source.size; cursor += 1){
    if(source.data[cursor] == '\n'){
      point_count += 1;
    }
  }

  Points result = points_allocate(allocator, point_count);

  point_count = 0;
  for(I64 cursor = 0; cursor < source.size;){
    I64 x0 = cursor;
    while(cursor < source.size && source.data[cursor] != ' '){ cursor += 1; }
    I64 x1 = cursor;
    cursor += 1;
    I64 y0 = cursor;
    while(cursor < source.size && source.data[cursor] != ' '){ cursor += 1; }
    I64 y1 = cursor;
    cursor += 1;
    I64 z0 = cursor;
    while(cursor < source.size && source.data[cursor] != '\n'){ cursor += 1; }
    I64 z1 = cursor;
    cursor += 1;

    F64 x, y, z;
    if(0
      || !parse_base10_string_as_f64(substring(source, x0, x1), &x)
      || !parse_base10_string_as_f64(substring(source, y0, y1), &y)
      || !parse_base10_string_as_f64(substring(source, z0, z1), &z))
    {
      panic("Failed to parse coordinate");
    }

    result.x[point_count] = x;
    result.y[point_count] = y;
    result.z[point_count] = z;
    point_count += 1;
  }

  return result;
}

#define QSORT_NAME sort_points_by_capacity
#define QSORT_TYPE Points
#define QSORT_COMPARE(_userdata_, _a_, _b_) ((_a_).count > (_b_).count)
#include "shared/qsort.inl"

typedef struct Trial Trial;
struct Trial{
  U64 building_us;
  U64 counting_us;
  U64 total_us;
};
DECLARE_ARRAY_TYPE(Trial);

typedef struct ProfileResult ProfileResult;
struct ProfileResult{
  Array(Trial) trials;
  F64 avg_total_ms;
  F64 std_total_ms;
  F64 avg_building_ms;
  F64 avg_counting_ms;
  U32 pair_count;
};

static ProfileResult profile_points(Points points, ThreadPool *pool, Allocator *allocator){
  ProfileResult result = {0};
  result.trials = array_create(Trial, allocator);

#define MAX_N_TRIALS 8192

  u64 total_time_start_us = os_get_monotonic_time_us();
  while(result.trials.count < MAX_N_TRIALS){
    U64 start_time_us = os_get_monotonic_time_us();
    #if APPROACH == APPROACH_kd_tree
    CalculateResult calc = calculate_pair_count_with_kd_tree(pool, points);
    #elif APPROACH == APPROACH_range_tree
    CalculateResult calc = calculate_pair_count_with_range_tree(pool, points);
    #elif APPROACH_gpu_grid
    CalculateResult calc = calculate_pair_count_with_gpu_grid(points);
    #else
    #error "Unknown approach"
    #endif
    U64 end_time_us = os_get_monotonic_time_us();

    if(result.trials.count && calc.pair_count != result.pair_count){
      panic("Got inconsitent pair count reported! Previously got %u pairs, now got %u pairs", result.pair_count, calc.pair_count);
    }
    result.pair_count = calc.pair_count;

    Trial trial = {0};
    trial.building_us = calc.building_us;
    trial.counting_us = calc.counting_us;
    trial.total_us = end_time_us - start_time_us;
    array_push(&result.trials, trial);

    if(os_get_monotonic_time_us() - total_time_start_us > 2000000) break;
  }

  U64 total_time_us = 0;
  U64 total_building_us = 0;
  U64 total_counting_us = 0;
  fiz(result.trials.count){
    total_time_us += result.trials.e[i].total_us;
    total_building_us += result.trials.e[i].building_us;
    total_counting_us += result.trials.e[i].counting_us;
  }
  U64 avg_us = total_time_us / result.trials.count;
  U64 avg_building_us = total_building_us / result.trials.count;
  U64 avg_counting_us = total_counting_us / result.trials.count;
  U64 var_us = 0;
  fiz(result.trials.count){
    var_us += SQUARE(avg_us - result.trials.e[i].total_us);
  }
  if(result.trials.count > 1){
    var_us /= (result.trials.count - 1);
  }

  result.avg_total_ms = avg_us / 1000.0;
  result.std_total_ms = sqrt((F64)var_us) / 1000.0;
  result.avg_building_ms = avg_building_us / 1000.0;
  result.avg_counting_ms = avg_counting_us / 1000.0;
  fprintf(stderr, "Found number of pairs in %.3f+-%.3fms (%.3fms build --> %.3fms count) (#trials=%d). Got %u pairs. \n",
                   result.avg_total_ms, result.std_total_ms,
                   result.avg_building_ms,
                   result.avg_counting_ms,
                   result.trials.count, result.pair_count);

  return result;
}

#if 0
static void test(){
  for(I64 index = 1; index < LIT_I64(100000000000000000); index <<= 1){
    U32 i = (U32)floorf( 0.5f + 0.5f * sqrtf((F32)(1 + 8 * index)) );
    U32 j = (U32)floor( 0.5 + 0.5 * sqrt((F64)(1 + 8 * index)) );
    if(i != j){
      fprintf(stderr, "Breakpoint at %li\n", index);
      return;
    }
  }
  fprintf(stderr, "No breakpoint\n");
}
#endif

int main(int argc, char **argv){
  // test();

  double rdtsc_to_us;
  {
    fprintf(stderr, "measuring rdtsc ...");
    I64 start_tsc = __rdtsc();
    U64 start_us = os_get_monotonic_time_us();
    os_sleep_us(1000);
    I64 end_tsc = __rdtsc();
    U64 end_us = os_get_monotonic_time_us();
    rdtsc_to_us = (end_us - start_us) / (double)(end_tsc - start_tsc);
    fprintf(stderr, "... measured rdtsc_to_us to %.9f us/rdtsc\n", rdtsc_to_us);
  }

  SpallProfile spall = {};
  #if USE_SPALL
  fprintf(stderr, "Using Spall to profile our application!\n");
  spall = SpallInit("out.spall", rdtsc_to_us);
  #else
  fprintf(stderr, "Not using spall!\n");
  #endif

  thread_context_init(&spall);

  #if CUDA_ENABLED
  {
    fprintf(stderr, "Running with CUDA enabled!\n");
    void *handle = dlopen("build/kernel.so", RTLD_LAZY); // Load CUDA code
    if(!handle){
      panic("Unable to open build/kernel.so!!");
    }
    GPU_GetAPIProc GPU_get_api_proc = dlsym(handle, "GPU_get_api");
    if(!GPU_get_api_proc){
      panic("Unable to load 'GPU_get_api' from build/kernel.so!!");
    }
    global_gpu_api = GPU_get_api_proc();

    global_gpu_api.init(&global_gpu_context, global_thread_context); // Make sure CUDA is initialized before we start doing measurements.
  }
  #else
  fprintf(stderr, "Running with CUDA disabled!\n");
  #endif

  Allocator *allocator = heap_allocator_make(NULL);

  ThreadPool *pool = thread_pool_make(allocator, &spall);

#define TEST_MULTIPLE_POINT_COUNTS 0

#if TEST_MULTIPLE_POINT_COUNTS
  {
    Array(Points) point_sets = array_create(Points, allocator);

    DIR *dir = opendir("data/");
    if(!dir){panic("Unable to open data/ directory");}
    for(struct dirent *ent; (ent = readdir(dir));){
      String name = string_from_cstring(ent->d_name);
      if(string_equals(name, LIT_STR(".")) || string_equals(name, LIT_STR(".."))){
        continue;
      }

      String path = allocator_push_printf(allocator, "data/%.*s", StrFormatArg(name));

      Points ps = parse_points(allocator, path);

      array_push(&point_sets, ps);
    }
    closedir(dir); dir = NULL;

    sort_points_by_capacity(NULL, point_sets.e, point_sets.count, allocator);

    StringBuilder out_csv = sb_create(allocator);

    for(i64 point_set_index = 0; point_set_index < point_sets.count; point_set_index += 1){
      Points points = point_sets.e[point_set_index];
      fprintf(stderr, "--> %u points\n", points.count);
      ProfileResult prof = profile_points(points, pool, allocator);

      if(point_set_index == 0){
        sb_printf(&out_csv, "#trials\t#points\tavg ms\tstd ms\t#pairs\n");
      }
      sb_printf(&out_csv, "%u\t%u\t%f\t%f\t%u\n", (U32)prof.trials.count, (U32)points.count, prof.avg_total_ms, prof.std_total_ms, (U32)prof.pair_count);
    }

    if(!dump_string_to_file(LIT_STR("out.csv"), sb_to_string(&out_csv, allocator))){
      panic("Unable to write out.csv");
    }
  }
#else // TEST_MULITPLE_POINT_COUNTS
  {
    U64 start_time_us = os_get_monotonic_time_us();
    Points points = parse_points(allocator, LIT_STR("positions.xyz"));
    U64 end_time_us = os_get_monotonic_time_us();
    fprintf(stderr, "Parsed everything in %.3fms. Got %u points.\n", (end_time_us - start_time_us) / 1000.0, points.count);
    profile_points(points, pool, allocator);
  }
#endif // TEST_MULITPLE_POINT_COUNTS

  thread_pool_destroy(pool); pool = NULL;

  allocator_destroy(&allocator);

  thread_context_destroy();
  #if USE_SPALL
  SpallQuit(&spall);
  #endif
  return 0;
}
