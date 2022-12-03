/*
hanna_util.h - Utility code for most of my projects in C!

Created: 2020-11-27
Author: hanna
License: Public Domain. See bottom of this file.

         The PCG PRNG code included here has license:
         *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
         Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

Essentially a single header library but all procedures are defined as static;
there is no HK_UTIL_IMPLEMENTATION macro as in most single header libraries.
*/

#ifndef HK_UTIL_H
#define HK_UTIL_H

#define STB_SPRINTF_IMPLEMENTATION
#include "3rdparty/stb_sprintf.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

//~ Sane names for data types
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;

typedef float f32;
typedef double f64;

// Alternative type names

typedef u64 U64;
typedef u32 U32;
typedef u16 U16;
typedef u8 U8;
typedef i64 I64;
typedef i32 I32;
typedef i16 I16;
typedef i8 I8;
typedef f64 F64;
typedef f32 F32;

// Literals

#define LIT_U64(_value_) ( _value_ ## ULL )
#define LIT_I64(_value_) ( _value_ ## LL )
#define LIT_U32(_value_) ( _value_ ## U )
#define LIT_I32(_value_) ( _value_ )

//~ Sane #defines

#define _CT_ASSERT_NAME2(_counter_) compile_time_assert##_counter_
#define _CT_ASSERT_NAME(_counter_) _CT_ASSERT_NAME2(_counter_)
#define CT_ASSERT(_condition_) \
  typedef struct{ int x[(_condition_) ? 1 : -1]; } _CT_ASSERT_NAME(__COUNTER__)
#define COMPILE_TIME_ASSERTION(...) CT_ASSERT(__VA_ARGS__)

#define _STRINGIZE(_a_) #_a_
#define STRINGIZE(_a_) _STRINGIZE(_a_)
#define _COMBINE2(_a_, _b_) _a_##_b_
#define COMBINE2(_a_, _b_) _COMBINE2(_a_, _b_)
#define COMBINE3(_a_, _b_, _c_) COMBINE2(_a_, COMBINE2(_b_, _c_))
#define COMBINE4(_a_, _b_, _c_, _d_) COMBINE2(_a_, COMBINE3(_b_, _c_, _d_))

#define IS_POWER_OF_TWO(_value_) ((((_value_) - 1) & (_value_)) == 0)

#define SIGN_OF_(_x_) ( ((_x_) < 0) ? (-1) : ( ((_x_) > 0) ? (1) : (0) ) )
#define ABSOLUTE_VALUE(_x_) ( ((_x_) < 0) ? (-(_x_)) : (((_x_) == 0) ? 0 : (_x_)) )
#define CLAMP(_a_, _x_, _b_) ( (_x_) < (_a_) ? (_a_) : ( (_x_) < (_b_) ? (_x_) : (_b_) ) )
#define CLAMP01(_x_) CLAMP(0, (_x_), 1)

// NOTE(hanna): These macros are copied from Shawn McGrath from when he used to stream. Thanks!
#define fiz(_count_) for(i64 i = 0; i < (_count_); ++i)
#define fjz(_count_) for(i64 j = 0; j < (_count_); ++j)
#define fkz(_count_) for(i64 k = 0; k < (_count_); ++k)

#define array_count(_array_) (sizeof(_array_) / sizeof((_array_)[0]))

#define MATH_PI (3.141592653589793238462643383279502884)
#define MATH_E (2.7182818284590452353602874713526624977572470936)
#define MATH_TAU (MATH_PI * 2.0)

#define DEG_TO_RAD(_value_) ((_value_) * (MATH_TAU / 360.0))
#define RAD_TO_DEG(_value_) ((_value_) * (360.0 / MATH_TAU))
#define SQUARE(_value_) ((_value_) * (_value_))

#define MINIMUM(_a_, _b_) ((_a_) < (_b_) ? (_a_) : (_b_))
#define MINIMUM3(_a_, _b_, _c_) (MINIMUM((_a_), MINIMUM((_b_), (_c_))))
#define MAXIMUM(_a_, _b_) ((_a_) > (_b_) ? (_a_) : (_b_))
#define MAXIMUM3(_a_, _b_, _c_) (MAXIMUM((_a_), MAXIMUM((_b_), (_c_))))

// Intended usage is IN_RANGE(lower <=, middle, <= upper) to check if lower <= middle <= upper
#define IN_RANGE(_a_, _b_, _c_) (_a_ _b_ && _b_ _c_)

#define KILOBYTES(_value_) ((_value_) * LIT_U64(1024))
#define MEGABYTES(_value_) ((_value_) * LIT_U64(1024) * LIT_U64(1024))
#define GIGABYTES(_value_) ((_value_) * LIT_U64(1024) * LIT_U64(1024) * LIT_U64(1024))

//~ Logging, panic and assertions.

__attribute__((__noreturn__)) static void _panic(const char *file, int line, const char *procedure_signature, const char *format, ...){
  va_list list;
  va_start(list, format);

  char buffer[4096];
  stbsp_vsnprintf(buffer, sizeof(buffer), format, list);
  fprintf(stderr, "PANIC on %s:%d (%s)! FATAL ERROR: %s\n", file, line, procedure_signature, buffer);

  va_end(list);

  __asm__("int3");
  exit(1);
}
#define panic(...) _panic(__FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)

#if DEBUG_BUILD

static int assertion_failed(const char *condition, int line, const char *file){
  fprintf(stderr, "(%s:%d) Assertion %s failed\n", file, line, condition);
  __asm__("int3");
  return 0;
}

#undef assert
#define assert(_condition_) ((_condition_) ? 0 : assertion_failed(#_condition_, __LINE__, __FILE__))
#else
#undef assert
#define assert(_condition_) ((_condition_), 0)
#endif

static void copy_memory(void *dst, size_t dst_size,
                        void *src, size_t src_size)
{
  assert(dst_size == src_size);
  memcpy(dst, src, dst_size);
}

#define zero_memory(_x_, _size_) memset((_x_), 0, (_size_))
#define zero_item(_x_) zero_memory((_x_), sizeof(*(_x_)))

static bool memory_equals(void *a, size_t size_a, void *b, size_t size_b){
  return (size_a == size_b) && (memcmp(a, b, size_a) == 0);
}

// ======================
// Basic OS interaction
// ======================

#if __linux__
#include <pthread.h>
typedef struct OSMutex{ pthread_mutex_t value; } OSMutex;
#else
#error "Unknown OS"
#endif
static void mutex_init(OSMutex *mutex);
static void mutex_destroy(OSMutex *mutex);
static void mutex_lock(OSMutex *mutex);
static void mutex_unlock(OSMutex *mutex);

#if __linux__
#include <semaphore.h>
typedef struct Semaphore{ sem_t value; } Semaphore;
#else
#error "Unknown OS"
#endif
static void semaphore_init(Semaphore *sem);
static void semaphore_destroy(Semaphore *sem);
static void semaphore_post(Semaphore *sem);
static void semaphore_wait(Semaphore *sem);
static bool semaphore_trywait(Semaphore *sem);
static bool semaphore_timedwait_ns(Semaphore *sem, u64 duration);
static int semaphore_get_value(Semaphore *sem);

#if __linux__
#undef _GNU_SOURCE
#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <pthread.h>

#include <unistd.h>
#include <time.h>
#include <dirent.h>

#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#include <stdio.h>
#include <errno.h>

//~ OSMutex

// TODO: We should make some proper comparison with using a Handmade Hero-style ticket mutex (perhaps use ips-student-challenge as a benchmark)

static void mutex_init(OSMutex *mutex){
  if(pthread_mutex_init(&mutex->value, NULL) != 0){
    panic("pthread_mutex_init() returned non-zero: errno = %d", errno);
  }
}
static void mutex_destroy(OSMutex *mutex){
  if(pthread_mutex_destroy(&mutex->value) != 0){
    panic("pthreaod_mutex_destroy() returned non-zero: errno = %d", errno);
  }
}
static void mutex_lock(OSMutex *mutex){
  if(pthread_mutex_lock(&mutex->value) != 0){
    panic("pthread_mutex_lock() returned non-zero: errno = %d", errno);
  }
}
static void mutex_unlock(OSMutex *mutex){
  if(pthread_mutex_unlock(&mutex->value) != 0){
    panic("pthread_mutex_unlock() returned non-zero: errno = %d", errno);
  }
}

// @TODO @IMPORTANT: Handle EINTR here
static void semaphore_init(Semaphore *sem){
  if(sem_init(&sem->value, 0, 0) == -1){ // does not fail with EINTR
    panic("sem_init() returned -1: errno = %d", errno);
  }
}
static void semaphore_destroy(Semaphore *sem){
  if(sem_destroy(&sem->value) == -1){ // does not fail with EINTR
    panic("sem_destroy() returned -1: errno = %d", errno);
  }
}
static void semaphore_post(Semaphore *sem){
  if(sem_post(&sem->value) == -1){ // does not fail with EINTR
    panic("sem_post() returned -1: errno = %d", errno);
  }
}
static void semaphore_wait(Semaphore *sem){
  int status;
  while((status = sem_wait(&sem->value)) == -1 && errno == EINTR);
  if(status == -1){
    panic("sem_wait() returned -1: errno = %d", errno);
  }
}
static bool semaphore_trywait(Semaphore *sem){
  bool result = true;
  int status;
  while((status = sem_trywait(&sem->value)) == -1 && errno == EINTR);
  int errno_value = errno;
  if(status == -1){
    if(errno_value == EAGAIN){
      result = false;
    }else{
      panic("sem_trywait() returned -1: errno = %d", errno_value);
    }
  }
  return result;
}
static bool semaphore_timedwait_ns(Semaphore *sem, u64 duration){
  bool result = true;
  // TODO: Verify that this actually works!!
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  u64 absolute_timeout = (uint64_t)ts.tv_sec * LIT_U64(1000000000) + (uint64_t)ts.tv_nsec;
  absolute_timeout += duration;
  struct timespec sleep_time = (struct timespec){
    .tv_sec = (i64)(absolute_timeout / LIT_U64(1000000000)),
    .tv_nsec = (i64)(absolute_timeout % LIT_U64(1000000000))
  };
  int status;
  while((status = sem_timedwait(&sem->value, &sleep_time)) == -1 && errno == EINTR);
  int errno_value = errno;
  if(status == -1){
    if(errno_value == ETIMEDOUT || errno_value == EAGAIN){
      result = false;
    }else{
      panic("sem_timedwait() returned -1: errno = %d", errno_value);
    }
  }
  return result;
}
static int semaphore_get_value(Semaphore *sem){
  int result;
  if(sem_getvalue(&sem->value, &result) == -1){
    panic("sem_getvalue() returned -1: errno = %d", errno);
  }
  if(result < 0){
    panic("sem_getvalue() gave back a negative semaphore value. We don't support systems with this behaviour. (But we trivially could by clamping the value.)");
  }
  return result;
}
#else
#error "Unknown OS"
#endif

// ==================
// GENERAL UTILITY
// ==================

//~ Atomics

// https://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync
// https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html

// NOTE: Atomics may not be placed across cache lines!!
typedef struct __attribute__((aligned(4))) AtomicU32{ u32 _value; } AtomicU32;
static u32 atomic_read_u32(AtomicU32 *atomic){ return __atomic_load_4(&atomic->_value, __ATOMIC_SEQ_CST); }
static void atomic_store_u32(AtomicU32 *atomic, u32 value){ __atomic_store_4(&atomic->_value, value, __ATOMIC_SEQ_CST); }
static u32 atomic_add_u32(AtomicU32 *atomic, u32 value){ return __atomic_fetch_add(&atomic->_value, value, __ATOMIC_SEQ_CST); }
static u32 atomic_sub_u32(AtomicU32 *atomic, u32 value){ return __atomic_fetch_sub(&atomic->_value, value, __ATOMIC_SEQ_CST); }
static bool atomic_compare_exchange_u32(AtomicU32 *atomic, u32 old_value, u32 new_value){
  return __atomic_compare_exchange_4(&atomic->_value,
                                     &old_value,
                                     new_value,
                                     /*weak=*/false,
                                     /*success memorder*/__ATOMIC_SEQ_CST,
                                     /*failure memorder*/__ATOMIC_SEQ_CST);
}

// NOTE: Atomics may not be placed across cache lines!!
typedef struct __attribute__((aligned(8))) AtomicU64{ u64 _value; } AtomicU64;
static u64 atomic_read_u64(AtomicU64 *atomic){ return __atomic_load_8(&atomic->_value, __ATOMIC_SEQ_CST); }
static void atomic_store_u64(AtomicU64 *atomic, u64 value){ __atomic_store_8(&atomic->_value, value, __ATOMIC_SEQ_CST); }
static u64 atomic_add_u64(AtomicU64 *atomic, u64 value){ return __atomic_fetch_add(&atomic->_value, value, __ATOMIC_SEQ_CST); }
static u64 atomic_sub_u64(AtomicU64 *atomic, u64 value){ return __atomic_fetch_sub(&atomic->_value, value, __ATOMIC_SEQ_CST); }
static bool atomic_compare_exchange_u64(AtomicU64 *atomic, u64 old_value, u64 new_value){
  return __atomic_compare_exchange_8(&atomic->_value,
                                     &old_value,
                                     new_value,
                                     /*weak=*/false,
                                     /*success memorder*/__ATOMIC_SEQ_CST,
                                     /*failure memorder*/__ATOMIC_SEQ_CST);
}

//
// STRINGS
//

// `String` = UTF-8 encoded immutable string; `Buffer` is just immutable raw non-textual data
typedef struct String String, Buffer;
struct String{
  u8 *data;
  size_t size;
};
CT_ASSERT(sizeof(String) == 16);

#define LitStr(_CString_) ((String){ .data = (u8*)(_CString_), .size = sizeof(_CString_) - 1 })
#define LIT_STR(_CString_) ((String){ .data = (u8*)(_CString_), .size = sizeof(_CString_) - 1 })

#define StrFormatArg(_String_) ((int)(_String_).size), ((char*)(_String_).data)

static String string_from_cstring(const char *cstring){
  String result = {0};
  result.data = (u8*)cstring;
  size_t size = 0;
  while(*cstring++){ size += 1; }
  result.size = size;
  return result;
}
static bool string_to_c_string(String string, char *buffer, size_t size){
  bool result = false;

  if(string.size + 1 <= size){
    result = true;
    memcpy(buffer, string.data, string.size);
    buffer[string.size] = '\0';
  }

  return result;
}

static String local_printf(u8 *buffer, size_t size, const char *format, ...){
  va_list list;
  va_start(list, format);
  assert(size < INT32_MAX);
  stbsp_vsnprintf((char*)buffer, (int)size, format, list);
  va_end(list);
  String result = string_from_cstring((char*)buffer);
  return result;
}
#define LOCAL_PRINTF(_name_, _size_, ...) u8 COMBINE2(_name_, __buffer)[(_size_)]; String _name_ = local_printf(COMBINE2(_name_, __buffer), sizeof(COMBINE2(_name_, __buffer)), __VA_ARGS__)

static String substring_nocheck(String string, u64 begin, u64 end){
  String result = {0};
  if(string.size >= end && end > begin){
    result.data = string.data + begin;
    result.size = (u32)(end - begin);
  }
  return result;
}
static String substring(String string, u64 begin, u64 end){
  assert(end >= begin);
  return substring_nocheck(string, begin, end);
}

static bool string_equals(String a, String b){
  bool result = false;
  if(a.data && b.data && a.size == b.size && a.size > 0){
    result = true;
    for(u64 index = 0; index < a.size; ++index){
      if(a.data[index] != b.data[index]){
        result = false;
        break;
      }
    }
  }
  return result;
}
static bool string_starts_with(String string, String with){
  return string_equals(substring(string, 0, with.size), with);
}

static bool string_consume(String *string, String with){
  bool result = string_starts_with(*string, with);
  if(result){
    *string = substring(*string, with.size, string->size);
  }
  return result;
}

static String string_strip_spaces(String value){
  String result = value;

  while(result.size > 0 && result.data[0] == ' '){
    result = substring(result, 1, result.size);
  }

  while(result.size > 0 && result.data[result.size - 1] == ' '){
    result = substring(result, 0, result.size - 1);
  }

  return result;
}

static bool split_string_in_two_at_ascii_character(String string, char ascii, String *before, String *after){
  bool result = false;
  u32 index = 0;
  for(; index < string.size; ++index){
    if(string.data[index] == ascii){
      break;
    }
  }

  if(index < string.size){
    result = true;

    *before = substring(string, 0, index);
    *after = substring(string, index + 1, string.size);
  }

  return result;
}

//~ UTF8 encoding/decoding

#define INVALID_CODEPOINT 0xffffffff

static bool is_utf8_continuation_byte(u8 value){
  bool result = false;
  if((value & 0xc0) == 0x80){
    result = true;
  }
  return result;
}

static u32 decode_utf8(u8 *input_begin, u8 *input_end, u32 *_encoded_size){
  u32 result = INVALID_CODEPOINT;
  u32 encoded_size = 0;

  if(input_begin < input_end){
    u8 value = input_begin[0];

    u8 masks[4]    = { 0x80, 0xe0, 0xf0, 0xf8 };
    u8 patterns[4] = { 0x00, 0xc0, 0xe0, 0xf0 };

    for(int index = 0; index < 4; ++index){
      if((value & masks[index]) == patterns[index]){
        if(input_begin + index + 1 <= input_end){
          encoded_size = index + 1;
        }
        break;
      }
    }

    if(encoded_size){
      result = value & ~masks[encoded_size - 1];

      for(int index = 1; index < encoded_size; ++index){
        result <<= 6;
        result |= input_begin[index] & 0x3f;
      }
    }
  }

  if(_encoded_size) *_encoded_size = encoded_size;
  return result;
}

// NOTE: Returns the number of bytes that were written to the buffer.
static u32 encode_utf8(u32 codepoint_init, u8 *buffer_begin, u8 *buffer_end){
  // TODO(hanna - 2019-03-22): This is highly untested! Use at your own risk!
  u32 result = 0;
  if(codepoint_init <= 0x7f && buffer_begin + 1 <= buffer_end){
    result = 1;
    buffer_begin[0] = codepoint_init;
  }else if(codepoint_init <= 0x7ff && buffer_begin + 2 <= buffer_end){
    result = 2;
  }else if(codepoint_init <= 0xffff && buffer_begin + 3 <= buffer_end){
    result = 3;
  }else if(codepoint_init <= 0x10ffff && buffer_begin + 4 <= buffer_end){
    result = 4;
  }

  if(result > 1){
    u32 codepoint = codepoint_init;
    buffer_begin[0] = 0x80;
    for(int index = result - 1; index > 0; --index){
      buffer_begin[0] |= 1 << (7 - index);
      buffer_begin[index] = 0x80 | (codepoint & 0x3f);
      codepoint >>= 6;
    }
    buffer_begin[0] |= codepoint;
  }

  return result;
}

//~ String decoding utility

// Peek codepoint in UTF-8 encoded string
static u32 peek_codepoint(String string, i64 cursor){
  return decode_utf8(string.data + cursor, string.data + string.size, NULL);
}
static bool next_codepoint(String string, i64 *_cursor){
  i64 cursor = *_cursor;

  bool result = false;
  if(cursor < string.size){
    result = true;

    u32 encoded_length;
    decode_utf8(string.data + cursor, string.data + string.size, &encoded_length);
    if(encoded_length){
      cursor += encoded_length;
    }else{
      cursor += 1;
    }
  }
  *_cursor = cursor;
  return result;
}
static bool prev_codepoint(String string, i64 *_cursor){
  i64 cursor = *_cursor;
  bool result = false;
  if(cursor > 0){
    result = true;
    cursor -= 1;
    while(cursor > 0 && is_utf8_continuation_byte(string.data[cursor])){
      cursor -= 1;
    }
  }
  *_cursor = cursor;
  return result;
}

//

/*
for(UTF8Iterator iter = iterate_utf8(string);
    iter.valid;
    advance_utf8_iterator(&iter))
{
}
*/
typedef struct UTF8Iterator UTF8Iterator;
struct UTF8Iterator{
  // INTERNALS
  u8 *begin;
  i64 at;
  u8 *end;

  // PUBLIC
  bool valid;
  i64 byte_index;
  i64 codepoint_index;
  u32 codepoint;
  u32 codepoint_bytes;
};
static UTF8Iterator iterate_utf8(String string);
static void advance_utf8_iterator(UTF8Iterator *iter);

static UTF8Iterator iterate_utf8(String string){
  UTF8Iterator result = {
    .begin = string.data,
    .end = string.data + string.size,
    .codepoint_index = -1,
  };
  advance_utf8_iterator(&result);
  return result;
}
static void advance_utf8_iterator(UTF8Iterator *iter){
  iter->codepoint = decode_utf8(iter->begin + iter->at, iter->end, &iter->codepoint_bytes);
  iter->byte_index = iter->at;
  iter->at += iter->codepoint_bytes;
  iter->valid = (iter->codepoint_bytes != 0);
  iter->codepoint_index += 1;
}

static i64 utf8_get_codepoint_count(String string){
  UTF8Iterator iter;
  for(iter = iterate_utf8(string);
      iter.valid;
      advance_utf8_iterator(&iter));
  return iter.codepoint_index;
}


#if 0
static void utf32_to_utf8(u32 *utf32, u32 utf32_count, String *_output){
  String output = *_output;

  fiz(utf32_count){
    u32 bytes = encode_utf8(utf32[i], output.data + output.size, output.data + output.capacity);
    if(bytes == 0){
      output = (String){0};
      break;
    }
    output.size += bytes;
  }

  *_output = output;
}
#endif

//~ String navigation for text editing UI

static i64 string_correct_cursor(String string, i64 pos){
  i64 result = CLAMP(0, pos, string.size);
  while(0 < result && result < string.size && is_utf8_continuation_byte(string.data[result])){
    result -= 1;
  }
  return result;
}

static i64 string_move_left(String string, i64 pos){
  i64 result = pos;

  if(0 < result && result <= string.size){
    result -= 1;
    while(0 < result && is_utf8_continuation_byte(string.data[result])){
      result -= 1;
    }
  }
  return result;
}
static i64 string_move_right(String string, i64 pos){
  i64 result = pos;

  if(0 <= result && result < string.size){
    result += 1;
    while(result < string.size && is_utf8_continuation_byte(string.data[result])){
      result += 1;
    }
  }

  return result;
}

static bool _string_move_is_whitespace(char c){
  return (c == ' ' || c == '\t');
}
static bool _string_move_is_word_char(char c){
  return ('a' <= c && c <= 'z')
      || ('A' <= c && c <= 'Z')
      || ('0' <= c && c <= '9')
      || (c == '_');
}
static i64 string_move_left_word(String string, i64 pos){
  i64 result = pos;

  if(0 < result && result <= string.size){
    while(0 < result && _string_move_is_whitespace(string.data[result - 1])){
      result -= 1;
    }

    while(0 < result && _string_move_is_word_char(string.data[result - 1])){
      result -= 1;
    }

    if(result == pos){
      result = pos - 1;
    }
  }

  return result;
}
static i64 string_move_right_word(String string, i64 pos){
  assert(0 <= pos);
  i64 result = pos;

  if(result < string.size){
    while(result < string.size && _string_move_is_whitespace(string.data[result])){
      result += 1;
    }

    while(result < string.size && _string_move_is_word_char(string.data[result])){
      result += 1;
    }

    if(result == pos){
      result = pos + 1;
    }
  }

  return result;
}

// *******************************************
//=============================================
// BEGIN OS LAYER CODE
//=============================================
// *******************************************

typedef struct OSFile{ u64 value; }OSFile;
typedef struct OSMappedFile{ void *data; i64 data_size; }OSMappedFile;
typedef struct OSThread{ u64 value; }OSThread;

#ifdef __x86_64__
#define OS_PAGE_SIZE (4096)
#else
#error "Unknown architecture"
#endif

static void* os_alloc_pages_commit(size_t size); // Returned pointer is NULL on failure; otherwise a pointer to commited zeroed pages is returned.
static void* os_alloc_pages_nocommit(size_t size); // Returned page is NULL on failure; otherwise a pointer to uncommited zeroed pages is returned.
static void os_free_pages(void *memory, size_t size);
static OSFile os_open_file_input(String path);
static OSFile os_open_file_output(String path);
static void os_close_file(OSFile file);
static i64 os_get_file_size(OSFile file);
static uint64_t os_get_file_modify_time_us(OSFile file); // In unix time in microseconds.
static bool os_read_from_file(OSFile file, i64 offset, u8 *buffer, u32 size);

/*
NOTE(hanna - 2020-01-26): The intended usage pattern of this is something like the following:

bool error = false;
uint64_t at = 0;
os_write_to_file(file, at, buffer, size, &error);
at += size;
...
os_write_to_file(file, at, buffer, size, &error);
at += size;
if(error){
// A write has failed!
// If 'error == true' then further calls to os_write_to_file are ignored.
}

If 'error = null' an attempt will always be made to perform the write, i.e. it has the same bahaviour as `*error = true`.
*/
static void os_write_to_file(OSFile file, i64 offset, u8 *buffer, u32 size, bool *error);

static OSMappedFile os_begin_memory_map_file_readonly(OSFile file);
static void os_end_memory_map_file(OSFile file, OSMappedFile mapped_file);

static u64 os_get_monotonic_time_us(); // NOTE: Both monotonically increasing & continous. Relative to an arbitrary point in time.
static u64 os_get_unix_time_us(); // NOTE: System time. Need not be continous.
static i64 os_get_unix_time();

typedef struct Allocator Allocator;
static String os_get_working_directory(Allocator *allocator);
static String os_get_home_directory(Allocator *allocator);

static OSThread os_start_thread(void (*entry_point)(void*), void *userdata, String name);
static void os_join_thread(OSThread thread);
static OSThread os_get_handle_to_current_thread();
static void os_sleep_us(uint64_t duration);

//=====================
// END PLATFORM CODE
//=====================

//
// MEMORY MANAGEMENT
//

/////////////////////////////////////////////////////////////////////////////
//~ Allocator interface (NOTE NOTE NOTE THAT THIS IS NOT MULTITHREAD SAFE!!!)

#include <setjmp.h>

typedef struct AllocatorBudget AllocatorBudget;
struct AllocatorBudget{
  size_t max_num_used_bytes;
  size_t used_bytes_watermark; // The highest `used_bytes` ever has been
  size_t used_bytes;

  bool oom_handler_set;
  jmp_buf oom_handler;
};

static void allocator_budget_signal_out_of_memory(AllocatorBudget *budget){
  assert(budget->oom_handler_set);
  longjmp(budget->oom_handler, 1);
}
static void allocator_budget_use_bytes(AllocatorBudget *budget, size_t bytes){
  assert(budget->oom_handler_set);
  if(budget->used_bytes + bytes > budget->max_num_used_bytes){
    // Memory budget exceeded!!!
    longjmp(budget->oom_handler, 1);
  }
  budget->used_bytes += bytes;
  budget->used_bytes_watermark = MAXIMUM(budget->used_bytes_watermark, budget->used_bytes);
}
static void allocator_budget_release_bytes(AllocatorBudget *budget, size_t bytes){
  //assert(budget->oom_handler_set); You may release bytes even when not in a begin/end pair
  assert(budget->used_bytes >= bytes && "Releasing more bytes than there are!!");
  budget->used_bytes -= bytes;
}
static void _allocator_budget_begin(AllocatorBudget *budget, size_t max_num_used_bytes){
  assert(!budget->oom_handler_set);
  budget->oom_handler_set = true;
  budget->max_num_used_bytes = max_num_used_bytes;
}
#define allocator_budget_begin(_budget_, _max_num_used_bytes_) \
  ( _allocator_budget_begin((_budget_), (_max_num_used_bytes_)), setjmp((_budget_)->oom_handler) ) \

static void allocator_budget_end(AllocatorBudget *budget){
  assert(budget->oom_handler_set);
  budget->oom_handler_set = false;
}


typedef struct AllocatorPageHeader AllocatorPageHeader;
struct AllocatorPageHeader{
  Allocator *allocator;
  AllocatorPageHeader *next;
  AllocatorPageHeader *prev;
  u8 _data[0];
};
CT_ASSERT(sizeof(AllocatorPageHeader) == 24); // just noting the size here
#define BIG_ALLOC_SIZE (OS_PAGE_SIZE - sizeof(AllocatorPageHeader) - sizeof(HeapAllocHeader) + 1)

static AllocatorPageHeader* get_allocator_page(void *alloc){
  assert(alloc);
  return (AllocatorPageHeader*)(((uintptr_t)alloc) & ~(uintptr_t)(OS_PAGE_SIZE - 1));
}
static Allocator* get_allocator(void *alloc){
  Allocator *result = get_allocator_page(alloc)->allocator;
  assert(result);
  return result;
}

typedef struct HeapAllocHeader HeapAllocHeader;
struct HeapAllocHeader{
  u16 next;
  u16 prev : 15;
  u16 occupied : 1;
  u8 data[0];
};
CT_ASSERT(sizeof(HeapAllocHeader) == 4);

static HeapAllocHeader* _heap_alloc_header_from_ptr(void *ptr){
  assert(ptr);
  HeapAllocHeader *result = (HeapAllocHeader*)((u8*)ptr - sizeof(HeapAllocHeader));
  return result;
}
static HeapAllocHeader* _heap_alloc_header_from_offset(AllocatorPageHeader *page, u16 offset){
  HeapAllocHeader *result = NULL;
  assert(offset >= sizeof(AllocatorPageHeader));
  if(offset < OS_PAGE_SIZE){
    result = (HeapAllocHeader*)((u8*)page + offset);
  }else{
    assert(offset == OS_PAGE_SIZE);
    result = (HeapAllocHeader*)page->_data;
  }
  return result;
}
static u16 _heap_alloc_header_offset(AllocatorPageHeader *page, HeapAllocHeader *alloc){
  return (u16)((uintptr_t)alloc - (uintptr_t)page);
}


typedef struct BigAllocHeader BigAllocHeader;
struct BigAllocHeader{
  AllocatorPageHeader page_header;
  size_t size; // size of `data`
  size_t total_size;
  u8 data[0];
};

typedef struct PushChunkHeader PushChunkHeader;
struct PushChunkHeader{
  AllocatorPageHeader page_header;
  u32 size;
  u8 data[0];
};

typedef struct Allocator Allocator;
struct Allocator{
  AllocatorBudget *budget;
  size_t num_used_bytes;

  //~ Alloc/Free interface
  AllocatorPageHeader heap_sentinel;
  BigAllocHeader big_alloc_sentinel;

  //~ Push allocator
  uintptr_t push_cursor;
  PushChunkHeader push_sentinel;

  //~ Stub
  AllocatorPageHeader *stub;
};

static AllocatorPageHeader* _allocator_alloc_pages(Allocator *allocator, AllocatorPageHeader *sentinel, size_t size){
  assert((size & (OS_PAGE_SIZE - 1)) == 0);

  if(allocator->budget){
    allocator_budget_use_bytes(allocator->budget, size);
  }
  AllocatorPageHeader *result = (AllocatorPageHeader*)os_alloc_pages_commit(size);
  if(!result){
    if(allocator->budget){
      allocator_budget_signal_out_of_memory(allocator->budget);
      // TODO: Log a warning?
    }else{
      panic("Allocator ran out of memory and there is no out of memory handler installed! Tried to allocate %I64u bytes.", (u64)size);
    }
  }
  allocator->num_used_bytes += size;

  result->allocator = allocator;

  if(sentinel){
    result->next = sentinel->next;
    result->next->prev = result;
    result->prev = sentinel;
    result->prev->next = result;
  }
  return result;
}
static void _allocator_free_pages_no_unlink(Allocator *allocator, AllocatorPageHeader *page, size_t size){
  os_free_pages(page, size);
  if(allocator->budget){
    allocator_budget_release_bytes(allocator->budget, size);
  }
}
static void _allocator_free_pages_unlink(Allocator *allocator, AllocatorPageHeader *page, size_t size){
  page->prev->next = page->next;
  page->next->prev = page->prev;
  _allocator_free_pages_no_unlink(allocator, page, size);
}

static void* _big_alloc_alloc(Allocator *allocator, size_t size, size_t align){
  size_t total_size = sizeof(BigAllocHeader) + size;
  total_size = (total_size + OS_PAGE_SIZE - 1) & ~(size_t)(OS_PAGE_SIZE - 1);

  BigAllocHeader *new_alloc = (BigAllocHeader*)_allocator_alloc_pages(allocator, &allocator->big_alloc_sentinel.page_header, total_size);

  // TODO VERY IMPORTANT: Handle alignment here.

  new_alloc->total_size = total_size;
  new_alloc->size = size;

  void *result = new_alloc->data;
  return result;
}

static void allocator_destroy(Allocator **_allocator){
  Allocator *allocator = *_allocator;
  *_allocator = NULL;
  if(allocator){
    for(AllocatorPageHeader *page = allocator->heap_sentinel.next; page != &allocator->heap_sentinel;){
      AllocatorPageHeader *next = page->next;
      _allocator_free_pages_no_unlink(allocator, page, OS_PAGE_SIZE);
      page = next;
    }

    for(BigAllocHeader *big_alloc = (BigAllocHeader*)allocator->big_alloc_sentinel.page_header.next; big_alloc != &allocator->big_alloc_sentinel;){
      BigAllocHeader *next = (BigAllocHeader*)big_alloc->page_header.next;
      _allocator_free_pages_no_unlink(allocator, &big_alloc->page_header, big_alloc->total_size);
      big_alloc = next;
    }

    for(PushChunkHeader *chunk = (PushChunkHeader*)allocator->push_sentinel.page_header.next; chunk != &allocator->push_sentinel;){
      PushChunkHeader *next = (PushChunkHeader*)chunk->page_header.next;
      _allocator_free_pages_no_unlink(allocator, &chunk->page_header, sizeof(PushChunkHeader) + chunk->size);
      chunk = next;
    }

    _allocator_free_pages_no_unlink(allocator, allocator->stub, OS_PAGE_SIZE);

    free(allocator);
  }
}

//~ Heap allocator
// This is a super naive heap allocator that is not designed to be efficient or good in any way.
// It was created because malloc didn't meet the requirements posed by the web server project.
// - hanna 2022-09-08

static size_t _heap_alloc_size(AllocatorPageHeader *page, HeapAllocHeader *alloc){
  assert(alloc->next >= _heap_alloc_header_offset(page, alloc));
  size_t result = alloc->next - _heap_alloc_header_offset(page, alloc) - sizeof(HeapAllocHeader);
  return result;
}
static void _heap_free(Allocator *allocator, void *ptr, size_t size);

static void* _heap_alloc(Allocator *allocator, size_t size, size_t align){
  // TODO TODO TODO: You should respect alignment here!

//#error "We should respect alignment here!"

  HeapAllocHeader *found_alloc = NULL;
  for(AllocatorPageHeader *page = allocator->heap_sentinel.next; page != &allocator->heap_sentinel; page = page->next){
    HeapAllocHeader *first_alloc = (HeapAllocHeader*)page->_data;
    HeapAllocHeader *test_alloc = first_alloc;
    do{
      HeapAllocHeader *next = _heap_alloc_header_from_offset(page, test_alloc->next);
      size_t test_size = _heap_alloc_size(page, test_alloc);
      if(!test_alloc->occupied && test_size >= sizeof(HeapAllocHeader) + size){
        found_alloc = test_alloc;
        break;
      }
      test_alloc = next;
    }while(test_alloc != first_alloc);
  }

  if(!found_alloc){
    AllocatorPageHeader *page = _allocator_alloc_pages(allocator, &allocator->heap_sentinel, OS_PAGE_SIZE);
    found_alloc = (HeapAllocHeader*)page->_data;
    found_alloc->prev = OS_PAGE_SIZE;
    found_alloc->next = OS_PAGE_SIZE;
  }

  AllocatorPageHeader *page = get_allocator_page(found_alloc);
  assert(_heap_alloc_size(page, found_alloc) >= sizeof(HeapAllocHeader) + size);

  found_alloc->occupied = 1;
  HeapAllocHeader *split_alloc = (HeapAllocHeader*)&found_alloc->data[size];
  *split_alloc = (HeapAllocHeader){
    .next = found_alloc->next,
    .prev = _heap_alloc_header_offset(page, found_alloc),
  };
  HeapAllocHeader *next = _heap_alloc_header_from_offset(page, found_alloc->next);
  next->prev = _heap_alloc_header_offset(page, split_alloc);

  found_alloc->next = _heap_alloc_header_offset(page, split_alloc);

  void *result = found_alloc->data;
  return result;
}

static void _heap_delete_alloc(AllocatorPageHeader *page, HeapAllocHeader *alloc){
  HeapAllocHeader *prev = _heap_alloc_header_from_offset(page, alloc->prev);
  HeapAllocHeader *next = _heap_alloc_header_from_offset(page, alloc->next);

  assert((void*)alloc != (void*)page->_data && "Cannot delete the first allocation");

  prev->next = alloc->next;
  next->prev = alloc->prev;
}

static void _heap_free(Allocator *allocator, void *ptr, size_t size){
  assert(ptr);

  AllocatorPageHeader *page = get_allocator_page(ptr);
  HeapAllocHeader *alloc = _heap_alloc_header_from_ptr(ptr);

  HeapAllocHeader *prev = _heap_alloc_header_from_offset(page, alloc->prev);
  HeapAllocHeader *next = _heap_alloc_header_from_offset(page, alloc->next);
  assert(_heap_alloc_size(page, alloc) == size);

  assert(alloc->occupied);
  alloc->occupied = 0;
  if(!prev->occupied && (uintptr_t)alloc != (uintptr_t)&page->_data){
    _heap_delete_alloc(page, alloc);
  }
  if(!next->occupied){
    _heap_delete_alloc(page, next);
  }
}

// NOTE: Destroyed with `allocator_destroy`
static Allocator* heap_allocator_make(AllocatorBudget *budget){
  Allocator *result = (Allocator*)calloc(1, sizeof(Allocator));
  result->budget = budget;
  result->heap_sentinel.next = result->heap_sentinel.prev = &result->heap_sentinel;
  result->big_alloc_sentinel.page_header.next = result->big_alloc_sentinel.page_header.prev = &result->big_alloc_sentinel.page_header;
  result->push_sentinel.page_header.next = result->push_sentinel.page_header.prev = &result->push_sentinel.page_header;
  result->stub = _allocator_alloc_pages(result, NULL, OS_PAGE_SIZE); // TODO: Check that none of the memory is tampered with?
  return result;
}

//~ Alloc/Free

static void* allocator_get_stub(Allocator *allocator){
  return allocator->stub->_data;
}

// Returns true if the allocation could be resized, false otherwise
static bool allocator_expand(void *ptr, size_t old_size, size_t new_size){
  assert(new_size > old_size);

  Allocator *allocator = get_allocator(ptr);

  bool result = false;
  if(ptr == allocator_get_stub(allocator)){
    // You can of course not expand the stub allocation.
    assert(old_size == 0);
  }else if(old_size < BIG_ALLOC_SIZE){
    // TODO: Make stuff happen here.
  }else{

  }
  return result;
}

static void allocator_free(void *ptr, size_t size){
  Allocator *allocator = get_allocator(ptr);

  if(ptr == allocator_get_stub(allocator)){
    assert(size == 0);
    // Do nothing!
  }else if(size < BIG_ALLOC_SIZE){
    _heap_free(allocator, ptr, size);
  }else{
    BigAllocHeader *page = (BigAllocHeader*)get_allocator_page(ptr);
    _allocator_free_pages_unlink(allocator, &page->page_header, page->total_size);
  }
}

static void allocator_realloc_noclear(void **_ptr, size_t old_size, size_t new_size, size_t align){
  assert(align != 0 && "Use align == 1 for no alignment requirements");
  assert((align & (align - 1)) == 0 && "Align must be a power of two!");
  // TODO put this assert back in:
  // assert(((uintptr_t)*_ptr & (uintptr_t)(align - 1)) == 0 && "Existing alloc must be properly aligned!");

  void *old_ptr = *_ptr;
  assert(old_ptr);
  Allocator *allocator = get_allocator(old_ptr);

  if(new_size < BIG_ALLOC_SIZE){
    *_ptr = _heap_alloc(allocator, new_size, align);
  }else{
    *_ptr = _big_alloc_alloc(allocator, new_size, align);
  }
  void *new_ptr = *_ptr;

  memcpy(new_ptr, old_ptr, MINIMUM(old_size, new_size));
  allocator_free(old_ptr, old_size);
}

static void* allocator_alloc_noclear(Allocator *allocator, size_t size, size_t align){
  void *result = allocator_get_stub(allocator);
  allocator_realloc_noclear(&result, 0, size, align);
  return result;
}
static void* allocator_alloc_clear(Allocator *allocator, size_t size, size_t align){
  void *result = allocator_alloc_noclear(allocator, size, align);
  memset(result, 0, size);
  return result;
}
#define allocator_alloc_item_noclear(_allocator_, _Type_) ( (_Type_*) allocator_alloc_noclear((_allocator_), sizeof(_Type_), __alignof__(_Type_)) )
#define allocator_alloc_item_clear(_allocator_, _Type_)   ( (_Type_*) allocator_alloc_clear  ((_allocator_), sizeof(_Type_), __alignof__(_Type_)) )

static String allocator_get_string_stub(Allocator *allocator){
  return (String){ .data = (u8*)allocator_get_stub(allocator) };
}
static String allocator_alloc_string(Allocator *allocator, String string){
  String result = {0};

  result.data = (u8*)allocator_alloc_noclear(allocator, string.size, 1);
  memcpy(result.data, string.data, string.size);
  result.size = string.size;

  return result;
}
static void allocator_realloc_string(String *dst, String src){
  Allocator *allocator = get_allocator(dst->data);
  allocator_free(dst->data, dst->size);
  *dst = (String){
    .data = (u8*)allocator_alloc_noclear(allocator, src.size, 1),
    .size = src.size,
  };
  memcpy(dst->data, src.data, src.size);
}

//~ Push allocation (Permanent allocations only freed when the allocator is destroyed)

static PushChunkHeader* _allocator_current_push_chunk(Allocator *allocator){
  return (PushChunkHeader*)allocator->push_sentinel.page_header.next;
}

static void* allocator_push_memory(Allocator *allocator, size_t element_size, size_t element_count, size_t align, bool zero){
  // TODO: Should we perhaps go into the out of memory path when these are this large?
  assert(element_size < UINT32_MAX && "Very large element size");
  assert(element_count < UINT32_MAX && "Very large element count");
  size_t size = element_size * element_count;

  assert(allocator);
  assert(align <= 64 && "Very large alignment");
  assert(((align - 1) & align) == 0 && "Align must be allocator power of two");

  allocator->push_cursor = ((allocator->push_cursor + align - 1) & ~(align - 1));

  if(!allocator->push_cursor || allocator->push_cursor + size > (uintptr_t)_allocator_current_push_chunk(allocator)->data + _allocator_current_push_chunk(allocator)->size){
    size_t chunk_size = MAXIMUM(KILOBYTES(16), size + sizeof(PushChunkHeader));
    chunk_size = (chunk_size + (size_t)(OS_PAGE_SIZE - 1)) & ~(size_t)(OS_PAGE_SIZE - 1);

    PushChunkHeader *chunk = (PushChunkHeader*)_allocator_alloc_pages(allocator, &allocator->push_sentinel.page_header, chunk_size);
    chunk->size = chunk_size - sizeof(PushChunkHeader);

    allocator->push_cursor = (uintptr_t)_allocator_current_push_chunk(allocator)->data;
    allocator->push_cursor = ((allocator->push_cursor + align - 1) & ~(align - 1));
  }

  assert((allocator->push_cursor & (align - 1)) == 0);
  assert(allocator->push_cursor + size <= (uintptr_t)_allocator_current_push_chunk(allocator)->data + _allocator_current_push_chunk(allocator)->size);

  void *result = (void*)allocator->push_cursor;
  allocator->push_cursor += size;
  return result;
}

#define allocator_push_items_noclear(_allocator_, _Type_, _n_) ( (_Type_*)allocator_push_memory((_allocator_), sizeof(_Type_), (size_t)(_n_), __alignof__(_Type_), false) )
#define allocator_push_items_clear(_allocator_, _Type_, _n_) ( (_Type_*)allocator_push_memory((_allocator_), sizeof(_Type_), (size_t)(_n_), __alignof__(_Type_), true) )
#define allocator_push_item_noclear(_allocator_, _Type_) allocator_push_items_noclear((_allocator_), _Type_, 1)
#define allocator_push_item_clear(_allocator_, _Type_) allocator_push_items_clear((_allocator_), _Type_, 1)

static char* allocator_push_string_as_cstring(Allocator *allocator, String string){
  char *result = allocator_push_items_noclear(allocator, char, string.size + 1);
  memcpy(result, string.data, string.size);
  result[string.size] = '\0';
  return result;
}

static u8* allocator_push_data(Allocator *allocator, u8 *data, size_t size){
  u8 *result = (u8*)allocator_push_items_noclear(allocator, u8, size);
  memcpy(result, data, size);
  return result;
}
static String allocator_push_string(Allocator *allocator, String string){
  String result = {0};
  if(string.size > 0){
    u8 *data = (u8*) allocator_push_items_noclear(allocator, u8, string.size);
    memcpy(data, string.data, string.size);
    result = (String){ .data = data, .size = string.size };
  }
  return result;
}

static String allocator_push_vprintf(Allocator *allocator, const char *format, va_list list1){
  va_list list2;
  va_copy(list2, list1);
  int size = stbsp_vsnprintf(NULL, 0, format, list1);
  String result = { .data = allocator_push_items_noclear(allocator, u8, size + 1), .size = (u32)size };
  stbsp_vsnprintf((char*)result.data, result.size + 1, format, list2);
  va_end(list2);
  return result;
}
static String allocator_push_printf(Allocator *allocator, const char *format, ...){
  va_list args;
  va_start(args, format);
  String result = allocator_push_vprintf(allocator, format, args);
  va_end(args);
  return result;
}

//=============================
// BEGIN LINUX IMPLEMENTATION
//=============================

#if __linux__

//
// NOTE: Include headers
//

#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <fcntl.h>
#include <dlfcn.h>

#include <unistd.h>
#include <time.h>
#include <dirent.h>

#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#include <stdio.h>
#include <errno.h>
#include <pthread.h>

//
// NOTE: Implementation of the API
//

static void *os_alloc_pages_commit(size_t size){
  void *result = NULL;
  result = mmap(NULL, size,
                PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE,
                -1, 0);
  if(result == (void*)-1){
    result = NULL;
  }
  return result;
}
static void *os_alloc_pages_nocommit(size_t size){
  void *result = NULL;
  result = mmap(NULL, size,
                PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE,
                -1, 0);
  if(result == (void*)-1){
    result = NULL;
  }
  return result;
}

static void os_free_pages(void *memory, size_t size){
  munmap(memory, size);
}

static OSFile os_open_file_input(String path){
  OSFile result = {0};
  char _path[4096];
  if(string_to_c_string(path, _path, sizeof(_path))){
    int fd = open(_path, O_RDONLY);
    result.value = (uint64_t)(fd + 1);
  }
  return result;
}
static OSFile os_open_file_output(String path){
  OSFile result = {0};
  char _path[4096];
  if(string_to_c_string(path, _path, sizeof(_path))){
    int fd = open(_path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    result.value = (uint64_t)(fd + 1);
  }
  return result;
}
static void os_close_file(OSFile file){
  if(file.value){
    int fd = (int)file.value - 1;
    close(fd);
  }
}
static i64 os_get_file_size(OSFile file){
  i64 result = 0;
  if(file.value){
    int fd = (int)file.value - 1;
    struct stat stat;
    fstat(fd, &stat);
    result = (i64)stat.st_size;
  }
  return result;
}
static uint64_t os_get_file_modify_time_us(OSFile file){
  uint64_t result = 0;
  if(file.value){
    int fd = (int)file.value - 1;
    struct stat stat;
    fstat(fd, &stat);
    struct timespec mtime = stat.st_mtim;
    result = (uint64_t)mtime.tv_sec * 1000000 + (uint64_t)mtime.tv_nsec / 1000;
  }
  return result;
}
static bool os_read_from_file(OSFile file, i64 offset, uint8_t *buffer, uint32_t size){
  bool result = false;
  if(file.value){
    int fd = (int)file.value - 1;
    ssize_t status = pread(fd, buffer, size, (off_t)offset);
    if(status == size){
      result = true;
    }
  }
  return result;
}
static void os_write_to_file(OSFile file, i64 offset, uint8_t *buffer, uint32_t size, bool *_error){
  bool error = false;
  if(_error){
    error = *_error;
  }

  if(error){
    // Error has already occured!
  }else if(!file.value){
    error = true;
  }else{
    int fd = (int)file.value - 1;
    ssize_t bytes_written = pwrite(fd, buffer, size, (off_t)offset);
    if((bytes_written == -1 || bytes_written < size)){
      error = true;
    }
  }

  if(_error){
    *_error = error;
  }
}

static OSMappedFile os_begin_memory_map_file_readonly(OSFile file){
  OSMappedFile result = {0};

  if(file.value){
    int fd = (int)file.value - 1;

    result.data_size = os_get_file_size(file);
    result.data = mmap(NULL, result.data_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if((uint64_t)result.data == (uint64_t)-1){
      result = (OSMappedFile){0};
    }
  }

  return result;
}
static void os_end_memory_map_file(OSFile file, OSMappedFile mapped_file){
  if(mapped_file.data){
    munmap(mapped_file.data, mapped_file.data_size);
  }
}

static uint64_t os_get_monotonic_time_us(){
  uint64_t result = 0;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  result = (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
  return result;
}
static uint64_t os_get_unix_time_us(){
  uint64_t result = 0;
  struct timeval tv = {0};
  gettimeofday(&tv, NULL);
  result = (uint64_t)tv.tv_sec * 1000000 + (uint64_t)tv.tv_usec;
  return result;
}
static i64 os_get_unix_time(){
  i64 result = 0;
  result = (i64)time(NULL);
  return result;
}

static String os_get_working_directory(Allocator *allocator){
  String result = {0};
  char buf[PATH_MAX];
  if(getcwd(buf, sizeof(buf))){
    result = allocator_push_string(allocator, string_from_cstring(buf));
  }
  return result;
}

static String os_get_home_directory(Allocator *allocator){
  String result = {0};
  char *home = getenv("HOME");
  if(home){
    result = allocator_push_string(allocator, string_from_cstring(home));
  }
  return result;
}

struct PThreadData{
  void (*entry_point)(void*);
  void *userdata;
};

static void* os_pthread_start_routine(void* _data){
  struct PThreadData *data = (struct PThreadData*)_data;
  data->entry_point(data->userdata);
  free(data);
  return 0;
}

static OSThread os_start_thread(void (*entry_point)(void*), void *userdata, String name){
  OSThread result = {0};

  struct PThreadData *data = (struct PThreadData*)malloc(sizeof(struct PThreadData));
  data->entry_point = entry_point;
  data->userdata = userdata;
  char _name[4096];

  if(!data || !string_to_c_string(name, _name, sizeof(_name))){
    // Error!
  }else{
    pthread_attr_t attr;
    if(pthread_attr_init(&attr) != 0){
      // Error!
    }else{
      pthread_t thread_id = 0;

      if(pthread_create(&thread_id, &attr, os_pthread_start_routine, data)){
        // Error!
      }else{
        extern int pthread_setname_np(pthread_t thread, const char *name); // Cannot get this declaration to show up in pthread.h so I put it here instead.
        pthread_setname_np(thread_id, _name);
        result.value = thread_id;
      }

      pthread_attr_destroy(&attr);
    }
  }

  if(result.value == 0){
    free(data);
  }

  return result;
}

static void os_join_thread(OSThread thread){
  pthread_t thread_id = thread.value;
  pthread_join(thread_id, NULL);
}

static OSThread os_get_handle_to_current_thread(){
  OSThread result = {0};
  result.value = pthread_self();
  return result;
}

static void os_sleep_us(uint64_t duration){
  struct timespec sleep_time = (struct timespec){
    .tv_sec = (i64)(duration / LIT_U64(1000000)),
    .tv_nsec = (i64)(1000 * (duration % LIT_U64(1000000)))
  };
  while(nanosleep(&sleep_time, &sleep_time) == -1 && errno == EINTR);
}

#else
#error "Unknown OS!"
#endif

//===============================
// END LINUX IMPLEMENTATION
//===============================

//
// OTHER UTILITY CODE
//

//~ Rolling hash

// NOTE(hanna - 2021-05-08): Some random prime number from the internet. I have no idea if it is good or
// not.
#define ROLLING_HASH_COEFFICIENT (4611686018427387631ULL)
typedef struct RollingHash RollingHash;
struct RollingHash{
  u8 *buffer;
  size_t buffer_size;
  u32 window_size;
  u64 coefficient_pow_window_size;
  // STATE
  u64 hash;
  i64 index;
};

static u64 rolling_hash_compute_hash(u8 *buffer, size_t buffer_size){
  u64 result = 0;
  fiz(buffer_size){
    result = result * ROLLING_HASH_COEFFICIENT + buffer[i];
  }
  return result;
}
static RollingHash rolling_hash_create(u8 *buffer, size_t buffer_size, u32 window_size){
  RollingHash result = {0};
  result.buffer = buffer;
  result.buffer_size = buffer_size;
  result.window_size = window_size;

  if(window_size <= buffer_size){
    result.coefficient_pow_window_size = 1;
    fiz(window_size){
      result.hash = result.hash * ROLLING_HASH_COEFFICIENT + buffer[i];
      result.coefficient_pow_window_size *= ROLLING_HASH_COEFFICIENT;
    }
  }
  return result;
}
static bool rolling_hash_is_valid(RollingHash *rh){
  return rh->index + rh->window_size <= rh->buffer_size;
}
static void rolling_hash_advance(RollingHash *rh){
  if(rh->index + rh->window_size < rh->buffer_size){
    rh->hash *= ROLLING_HASH_COEFFICIENT;
    rh->hash += rh->buffer[rh->index + rh->window_size];
    rh->hash -= rh->buffer[rh->index] * rh->coefficient_pow_window_size;
  }
  rh->index += 1;
}

//~ CSV loading code

#if 0

typedef struct CSVFile CSVFile;
struct CSVFile{
  Allocator *push_allocator;

  String file_content;
  u64 at;
  char separator_ascii_character;

  u32 column_count;
  String *column_names;

  u32 row_count;

  u32 one_past_current_row_number;
  String *current_row; // array of `column_count` values
};

static String read_line(String file_content, u64 *_at){
  assert(_at);
  u64 at = *_at;
  assert(at < file_content.size);

  u64 line_begin = at;
  while(at < file_content.size && file_content.data[at] != '\n'){
    at += 1;
  }
  u64 line_end = at;
  if(at < file_content.size && file_content.data[at] == '\n'){
    at += 1;
  }

  String result = substring(file_content, line_begin, line_end);
  *_at = at;
  return result;
}


static bool csv_file_read_row(CSVFile *file, String *_error_message){
  bool result = false;
  String error_message = {0};
  if(_error_message){
    error_message = *_error_message;
  }

  if(file->at < file->file_content.size){
    result = true;
    file->one_past_current_row_number += 1;
    String line = read_line(file->file_content, &file->at);

    u64 cursor = 0;
    u64 num_lexed_fields = 0;

    while(cursor <= line.size){
      String token = {0};

      if(cursor < line.size && line.data[cursor] == '"'){
        cursor += 1;
        u64 token_begin = cursor;
        while(cursor < line.size && line.data[cursor] != '"'){
          cursor += 1;
        }

        u64 token_end = cursor;
        if(cursor == line.size){
          result = false;
          str_printf(&error_message, "Unterminated quote on row %d.", file->one_past_current_row_number - 1);
          goto failure;
        }
        cursor += 1; // eat quote

        if(cursor + 1 < line.size && line.data[cursor + 1] != file->separator_ascii_character){
          result = false;
          str_printf(&error_message, "Expected separator character `%c` after double-quoted field on row %d.", file->separator_ascii_character, file->one_past_current_row_number - 1);
          goto failure;
        }
        cursor += 1; // skip separator character or skip past the end of the line

        token = substring(line, token_begin, token_end);
      }else{
        u64 token_begin = cursor;
        while(cursor < line.size && line.data[cursor] != file->separator_ascii_character){
          cursor += 1;
        }

        u64 token_end = cursor;
        cursor += 1;
        token = substring(line, token_begin, token_end);
      }

      if(num_lexed_fields < file->column_count){
        file->current_row[num_lexed_fields] = token;
        num_lexed_fields += 1;
      }else{
        result = false;
        str_printf(&error_message, "Too many entries on row %d.", file->one_past_current_row_number - 1);
        goto failure;
      }
    }failure:;

    if(result && num_lexed_fields < file->column_count){
      result = false;
      str_printf(&error_message, "Too few entries on row %d.", file->one_past_current_row_number - 1);
    }
  }

  if(_error_message){
    *_error_message = error_message;
  }

  return result;
}

static CSVFile csv_file_create(MemoryArena *arena, String file_content, char separator_ascii_character, u32 column_count, String *column_names, String *_error_message){
  String error_message = _error_message ? *_error_message : (String){0};

  CSVFile result = {0};
  result.arena = arena;
  result.file_content = file_content;
  result.separator_ascii_character = separator_ascii_character;
  result.column_count = column_count;
  result.column_names = column_names;

  result.current_row = arena_push_items_noclear(arena, String, column_count);

  if(csv_file_read_row(&result, &error_message)){
    fiz(result.column_count){
      String field = result.current_row[i];

      if(!string_equals(field, column_names[i])){
        str_printf(&error_message, "Header of file doesn't match given strings (expected `%*.s`, found `%*.s`)", column_names[i], field);
        break;
      }
    }

    if(error_message.size == 0){
      u64 at = result.at;

      while(at < file_content.size){
        read_line(file_content, &at);
        result.row_count += 1;
      }
    }
  }else{
    result = (CSVFile){0};
  }

  if(_error_message){
    *_error_message = error_message;
  }

  return result;
}


/*
NOTE(hanna - 2020-11-27):
The intended usage here is to to have an outer loop over each row (AKA record) of the CSV file and in this loop you use a pair
of these to process each of the fields to make sure that the fields are being processed in the right order. In this outer loop
you have a `field_cursor` which will be read and then incremented by these macros to process the fields one by one.

In code, this would look somewhat like this:

for(i64 row_cursor = 0; csv_file_read_row(&csv, &error_message); row_cursor += 1){
  i64 field_cursor = 0;

  CSV_BEGIN_FIELD("name"){
    names[row_cursor] = arena_push_string(arena, field_string);
  } CSV_END_FIELD();

  CSV_BEGIN_FIELD("cost"){
    i64 value;
    if(parse_i64(field_string, &value, 10)){
      costs[row_cursor] = value;
    }else{
      failure = true;
    }
  } CSV_END_FIELD();
}

*/
#define CSV_BEGIN_FIELD(_csv_, _name_) do{ assert((u8*)(_name_) == (_csv_)->column_names[field_cursor].data); String field_string = csv.current_row[field_cursor]; if(1)
#define CSV_END_FIELD(_csv_) do{ field_cursor += 1; } while(false); } while(false)

#define CSV_PARSE_F64_FIELD(_csv_, _name_, _array_, _error_message_) \
  do{ \
    CSV_BEGIN_FIELD(_csv_, _name_); \
      f64 value; \
      if(parse_base10_string_as_f64(field_string, &value)){ \
        (_array_)[row_cursor] = value;\
      }else{ \
        str_printf((_error_message_), \
                         "Unable to parse field `%s` on row %d. Unable to parse field value `%.*s` as base 10 "\
                         "decimal number.", (_name_), (_csv_)->one_past_current_row_number - 1, \
                         StrFormatArg(field_string)); \
      }\
    CSV_END_FIELD(_csv_); \
  }while(false)
#endif

//~ Parsing utility

static bool parse_i64(String string, i64 *_output, i64 base){
  bool result = false;
  i64 output = 0;

  if(string.size > 0){
    i64 i = 0;
    bool negative = false;
    if(string.data[i] == '-'){
      i += 1;
      negative = true;
    }

    // TODO: Overflow protection?

    result = true;
    for(; i < string.size; i+= 1){
      int digit = -1;
      if('0' <= string.data[i] && string.data[i] <= '9'){
        digit = string.data[i] - '0';
      }else if('a' <= string.data[i] && string.data[i] <= 'f'){
        digit = string.data[i] + 10 - 'a';
      }else if('A' <= string.data[i] && string.data[i] <= 'F'){
        digit = string.data[i] + 10 - 'A';
      }

      if(digit == -1 || digit >= base){
        result = false;
        output = 0;
        break;
      }

      output = output * base + digit;
    }

    if(negative){
      output = -output;
    }
  }

  *_output = output;
  return result;
}

#include <math.h>
#include <errno.h>
static bool parse_base10_string_as_f64(String string, f64 *_output){
  bool result = false;

  f64 output = 0;
  // TODO(hanna - 2020-11-21): Implement our own base10 string -> IEEE754 double precision floating point routine which handles bad input correctly.

  char c_string[256];

  if(string.size < sizeof(c_string)){
    result = true;

    memcpy(c_string, string.data, string.size);
    c_string[string.size] = '\0';

    errno = 0;
    char *end_pointer;
    output = strtod(c_string, &end_pointer);

    if(errno && end_pointer != c_string + string.size){
      result = false;
    }
  }

  if(_output){
    *_output = output;
  }

  return result;
}

//~ Basic lexer

// A very simplistic lexer built for very simplistic syntax highlighting.
typedef struct BasicLexer BasicLexer;
struct BasicLexer{
  String content;
  i64 cursor;

#define BASIC_TOKEN_KIND_whitespace         1
#define BASIC_TOKEN_KIND_identifier         2
#define BASIC_TOKEN_KIND_number             3
#define BASIC_TOKEN_KIND_string             4
#define BASIC_TOKEN_KIND_codepoint          5
#define BASIC_TOKEN_KIND_comment            6
  u32 token_kind;
  String token;
};
static bool _basic_lexer_is_ascii_letter(u32 codepoint){
  return ('a' <= codepoint && codepoint <= 'z') || ('A' <= codepoint && codepoint <= 'Z');
}
static bool _basic_lexer_is_ascii_digit(u32 codepoint){
  return ('0' <= codepoint && codepoint <= '9');
}
static bool _basic_lexer_is_ascii_horz_whitespace(u32 codepoint){
  return (codepoint == '\t' || codepoint == ' ');
}
static bool basic_lexer_next_token(BasicLexer *l){
  bool result = true;
  i64 begin = l->cursor;
  if(l->cursor >= l->content.size){ // Nothing left!
    result = false;
  }else if(_basic_lexer_is_ascii_horz_whitespace(l->content.data[l->cursor])){
    l->cursor += 1;
    l->token_kind = BASIC_TOKEN_KIND_whitespace;
    while(l->cursor < l->content.size && _basic_lexer_is_ascii_horz_whitespace(l->content.data[l->cursor])){
      l->cursor += 1;
    }
  }else if((_basic_lexer_is_ascii_letter(l->content.data[l->cursor]) || l->content.data[l->cursor] == '_')){ // Word!
    l->cursor += 1;
    l->token_kind = BASIC_TOKEN_KIND_identifier;
    while(l->cursor < l->content.size && (_basic_lexer_is_ascii_letter(l->content.data[l->cursor]) || l->content.data[l->cursor] == '_' || _basic_lexer_is_ascii_digit(l->content.data[l->cursor]))){
      l->cursor += 1;
    }
  }else if(_basic_lexer_is_ascii_digit(l->content.data[l->cursor])){ // Number!!
    l->cursor += 1;
    l->token_kind = BASIC_TOKEN_KIND_number;
    while(l->cursor < l->content.size && _basic_lexer_is_ascii_digit(l->content.data[l->cursor])){
      l->cursor += 1;
    }
  }else if(l->content.data[l->cursor] == '"'){
    l->cursor += 1;
    l->token_kind = BASIC_TOKEN_KIND_string;
    while(l->cursor < l->content.size && l->content.data[l->cursor] != '\n' && l->content.data[l->cursor] != '"'){
      if(l->content.data[l->cursor] == '\\'){
        l->cursor += 2;
      }else{
        l->cursor += 1;
      }
    }
    l->cursor += 1;
  }else if(l->cursor + 2 <= l->content.size && l->content.data[l->cursor + 0] == '/' && l->content.data[l->cursor + 1] == '/'){
    // A line comment
    l->token_kind = BASIC_TOKEN_KIND_comment;
    while(l->cursor < l->content.size && l->content.data[l->cursor] != '\n'){
      l->cursor += 1;
    }
    l->cursor += 1;
  }else{ // A stray codepoint
    l->token_kind = BASIC_TOKEN_KIND_codepoint;
    next_codepoint(l->content, &l->cursor);
  }
  i64 end = l->cursor;

  l->token = substring(l->content, begin, end);

  return result;
}

//~ Data vector utility

#if 0

static f64* f64vector_compute_squared_differances(MemoryArena *arena, i64 element_count, f64 *a, f64 *b){
  f64 *result = arena_push_items_noclear(arena, f64, element_count);
  fiz(element_count){
    result[i] = SQUARE(a[i] - b[i]);
  }
  return result;
}
static f64 f64vector_compute_sum(i64 element_count, f64 *values){
  f64 result = 0;
  fiz(element_count){
    result += values[i];
  }
  return result;
}
#endif


//~ Floating point

// TODO(hanna - 2021-02-15): Don't use the standard library!
#include <math.h>
static f32 f32_fractional_part(f32 x){
  f32 i;
  return modff(x, &i);
}

static void assert_f32_is_not_fishy(f32 x){
  assert(x == x);
  assert(x != INFINITY);
  assert(x != -INFINITY);
}

static f32 f32_mix(f32 a, f32 factor, f32 b){
  return a * (1 - factor) + b * factor;
}

static i32 fast_floor_f32_to_i32(f32 value){ // TODO: Think through this
  return (i32)value - (i32)(value < 0);
}

static bool f32_approx_equals(f32 a, f32 b, f32 epsilon){
  bool result = false;
  if(-epsilon < a - b && a - b < epsilon){
    result = true;
  }
  return result;
}

static f32 f32_absolute(f32 value){
  if(value < 0) return -value;
  return value;
}

#define F32_INFINITY INFINITY
#define F32_NAN (0.f / 0.f)

//~ Swapping

static void f32_swap(f32 *a, f32 *b){
  f32 tmp = *a;
  *a = *b;
  *b = tmp;
}
static void i64_swap(i64 *a, i64 *b){
  i64 tmp = *a;
  *a = *b;
  *b = tmp;
}
static void u64_swap(u64 *a, u64 *b){
  u64 tmp = *a;
  *a = *b;
  *b = tmp;
}
static void u8_swap(u8 *a, u8 *b){
  u8 tmp = *a;
  *a = *b;
  *b = tmp;
}

//~ Comparison

static int i64_compare(i64 a, i64 b){
  int result;
  if(a == b){
    result = 0;
  }else if(a > b){
    result = 1;
  }else{
    result = -1;
  }
  return result;
}
static int string_compare(String a, String b){
  int result = i64_compare(a.size, b.size);
  if(result == 0){
    result = memcmp(a.data, b.data, a.size);
  }
  return result;
}


//~ Bitset

typedef struct Bitset Bitset;
struct Bitset{
  u64 *bits;
  u64 num_bits;
};
static Bitset push_bitset(Allocator *allocator, u64 num_bits){
  Bitset result = {0};
  result.num_bits = num_bits;
  result.bits = allocator_push_items_clear(allocator, u64, (num_bits + 0x3f) >> 6);
  return result;
}
static bool bitset_get(Bitset *bitset, u64 index){
  assert(index < bitset->num_bits);
  bool result = false;
  u64 high_index = index >> 6;
  u64 low_index = index & 0x3f;
  if(bitset->bits[high_index] & ((u64)1 << low_index)){
    result = true;
  }
  return result;
}
static void bitset_set(Bitset *bitset, u64 index, bool value){
  assert(index < bitset->num_bits);
  u64 high_index = index >> 6;
  u64 low_index = index & 0x3f;
  bitset->bits[high_index] &= ~((u64)1 << low_index);
  bitset->bits[high_index] |= (u64)(!!value) << low_index;
}

//~ Bits

static u32 u32_bitswap(u32 value, u32 bit_a, u32 bit_b){
  u32 result = value;
  result &= ~((1U << bit_a) | (1U << bit_b));
  result |= ((value >> bit_a) & 1U) << bit_b;
  result |= ((value >> bit_b) & 1U) << bit_a;
  return result;
}

//~ Find bit positions

static u32 u32_get_lsb(u32 value){
  if(value){
    return __builtin_ctz(value);
  }else{
    return 32;
  }
}

//~ Bitwise conversion

static u64 f64_bitwise_as_u64(f64 value){
  union{
    u64 _u64;
    f64 _f64;
  } conversion = { ._f64 = value };
  return conversion._u64;
}
static f64 u64_bitwise_as_f64(u64 value){
  union{
    u64 _u64;
    f64 _f64;
  } conversion = { ._u64 = value };
  return conversion._f64;
}
static u64 i64_bitwise_as_u64(i64 value){
  union{
    u64 _u64;
    i64 _i64;
  } conversion = { ._i64 = value };
  return conversion._u64;
}
static i64 u64_bitwise_as_i64(u64 value){
  union{
    u64 _u64;
    i64 _i64;
  } conversion = { ._u64 = value };
  return conversion._i64;
}
static u32 i64_bitwise_as_u32(i32 value){
  union{
    u32 _u32;
    i32 _i32;
  } conversion = { ._i32 = value };
  return conversion._u32;
}
static i32 u32_bitwise_as_i32(u32 value){
  union{
    u32 _u32;
    i32 _i32;
  } conversion = { ._u32 = value };
  return conversion._i32;
}

//
// NOTE: Vector and matrix math
//

#include <smmintrin.h>
#include <immintrin.h>

typedef struct V2{ f32 x, y; } V2;
typedef struct V2i{ i32 x, y; } V2i;
typedef union V3{ struct{ f32 x, y, z; }; struct{ f32 r, g, b; }; f32 e[3]; } V3;
typedef struct V3i{ i32 x, y, z; } V3i;
typedef union V4{ struct{ f32 x, y, z, w; }; struct{ f32 r, g, b, a; }; struct{ V3 rgb; }; } V4;
typedef struct Mat4x4{ f32 e[4][4]; } Mat4x4;
#define MAT4x4_IDENTITY ((Mat4x4){ .e[0][0] = 1, .e[1][1] = 1, .e[2][2] = 1, .e[3][3] = 1 })

//~ V2

static V2 v2(f32 x, f32 y){ return (V2){ x, y }; }
static V2 vec2(f32 x, f32 y){ return (V2){ x, y }; }
static V2 v2_scalar_mul(V2 a, f32 scalar){ return (V2){ a.x * scalar, a.y * scalar }; }
static V2 v2_add(V2 a, V2 b){ return (V2){ a.x + b.x, a.y + b.y }; }
static V2 v2_sub(V2 a, V2 b){ return (V2){ a.x - b.x, a.y - b.y }; }
static f32 v2_dot(V2 a, V2 b){ return a.x * b.x + a.y * b.y; }
static V2 v2_mix(V2 a, f32 t, V2 b){ return v2_add( v2_scalar_mul(a, 1 - t), v2_scalar_mul(b, t) ); }
static V2 v2_negate(V2 a){ return (V2){ -a.x, -a.y }; }
static V2 v2_componentwise_div(V2 a, V2 b){ return (V2){ a.x / b.x, a.y / b.y }; }
static V2 v2_componentwise_mul(V2 a, V2 b){ return (V2){ a.x * b.x, a.y * b.y }; }
static V2 v2_hadamard(V2 a, V2 b){ return (V2){ a.x * b.x, a.y * b.y }; }
static f32 v2_distance_squared(V2 a, V2 b){ return SQUARE(a.x - b.x) + SQUARE(a.y - b.y); }
static f32 v2_length_squared(V2 v){ return SQUARE(v.x) + SQUARE(v.y); }
static V2 v2_normalize(V2 v){
  f32 inv_length = 1.f / sqrtf(SQUARE(v.x) + SQUARE(v.y));
  return v2(v.x * inv_length, v.y * inv_length);
}
static f32 v2_cross(V2 a, V2 b){ return a.x * b.y - a.y * b.x; }
static bool v2_bitwise_equal(V2 a, V2 b){ return (a.x == b.x && a.y == b.y); }

static V2i v2i(i32 x, i32 y){ return (V2i){ x, y }; }
static V2i v2i_scalar_mul(V2i a, i32 scalar){ return v2i(a.x * scalar, a.y * scalar); }
static V2i v2i_add(V2i a, V2i b){ return v2i(a.x + b.x, a.y + b.y); }
static V2i v2i_sub(V2i a, V2i b){ return v2i(a.x - b.x, a.y - b.y); }
static i32 v2i_dot(V2i a, V2i b){ return a.x * b.x + a.y * b.y; }
static V2i v2i_negate(V2i a){ return v2i(-a.x, -a.y); }

V2 v2_from_v2i(V2i v){ return v2((f32)v.x, (f32)v.y); }
V2i v2i_from_v2(V2 v){ return v2i((i32)v.x, (i32)v.y); }

//~ V3

static V3 v3(f32 x, f32 y, f32 z){ return (V3){ x, y, z }; }
static V3 vec3(f32 x, f32 y, f32 z){ return (V3){ x, y, z }; }
static V3 vec3_set1(f32 x){ return (V3){ x, x, x }; }

static V3 v3_scalar_mul(V3 a, f32 scalar){ return (V3){ a.x * scalar, a.y * scalar, a.z * scalar }; }
static V3 v3_add(V3 a, V3 b){ return (V3){ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 v3_sub(V3 a, V3 b){ return (V3){ a.x - b.x, a.y - b.y, a.z - b.z }; }
static f32 v3_dot(V3 a, V3 b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
// NOTE(hanna - 2020-09-18): We work in a right hand coordinate system.
static V3 v3_cross(V3 a, V3 b){ return (V3){ a.y * b.z - a.z * b.y,
                                             a.z * b.x - a.x * b.z,
                                             a.x * b.y - a.y * b.x }; }
static f32 v3_length_squared(V3 v){ return SQUARE(v.x) + SQUARE(v.y) + SQUARE(v.z); }
static f32 v3_distance_squared(V3 a, V3 b){ return SQUARE(a.x - b.x) + SQUARE(a.y - b.y) + SQUARE(a.z - b.z); }
static V3 v3_mix(V3 a, f32 factor, V3 b){
  return v3_add( v3_scalar_mul(a, 1 - factor), v3_scalar_mul(b, factor) );
}
static void v3_swap(V3 *a, V3 *b){
  V3 tmp = *a;
  *a = *b;
  *b = tmp;
}

static f32 v3_max_element(V3 v){ return MAXIMUM3(v.x, v.y, v.z); }

static V3 v3_hadamard(V3 a, V3 b){ return (V3){ a.x * b.x, a.y * b.y, a.z * b.z }; }

//~ V3i

static V3i v3i(i32 x, i32 y, i32 z){ return (V3i){ x, y, z }; }
static V3i v3i_scalar_mul(V3i a, i32 scalar){ return (V3i){ a.x * scalar, a.y * scalar, a.z * scalar }; }
static V3i v3i_add(V3i a, V3i b){ return (V3i){ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3i v3i_sub(V3i a, V3i b){ return (V3i){ a.x - b.x, a.y - b.y, a.z - b.z }; }
static i32 v3i_dot(V3i a, V3i b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
// NOTE(hanna - 2020-09-18): We work in a right hand coordinate system.
static V3i v3i_cross(V3i a, V3i b){ return (V3i){ a.y * b.z - a.z * b.y,
                                                  a.z * b.x - a.x * b.z,
                                                  a.x * b.y - a.y * b.x }; }

static V3i v3i_from_v3(V3 v){ return (V3i){ (i32)v.x, (i32)v.y, (i32)v.z }; }

//~ V4

static V4 v4(f32 x, f32 y, f32 z, f32 w){ return (V4){ x, y, z, w }; }
static V4 v4_scalar_mul(V4 a, f32 scalar){ return (V4){ a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar }; }
static V4 v4_add(V4 a, V4 b){ return (V4){ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
static V4 v4_sub(V4 a, V4 b){ return (V4){ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
static f32 v4_dot(V4 a, V4 b){ return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static V4 v4_mix(V4 a, f32 t, V4 b){ return v4_add( v4_scalar_mul(a, 1 - t), v4_scalar_mul(b, t) ); }
static __m128 v4_to_m128(V4 v){ return _mm_set_ps(v.w, v.z, v.y, v.x); }
static V4 m128_to_v4(__m128 v){ V4 result; _mm_store_ps((f32*)&result, v); return result; }
static bool v4_bitwise_equal(V4 a, V4 b){ return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

//~ Mat4x4

static Mat4x4 mat4x4_transpose(Mat4x4 value){
  Mat4x4 result = {0};
  for(int j = 0; j < 4; ++j) for(int i = 0; i < 4; ++i){
    result.e[i][j] = value.e[j][i];
  }
  return result;
}
static Mat4x4 mat4x4_mul(Mat4x4 a, Mat4x4 b){
  Mat4x4 result = {0};
  for(int j = 0; j < 4; ++j) for(int i = 0; i < 4; ++i){
    result.e[j][i] = a.e[j][0] * b.e[0][i]
                   + a.e[j][1] * b.e[1][i]
                   + a.e[j][2] * b.e[2][i]
                   + a.e[j][3] * b.e[3][i];
  }
  return result;
}
static V4 mul_mat4x4_v4(Mat4x4 m, V4 v){
  V4 result = {0};
  result.x = v.x * m.e[0][0] + v.y * m.e[0][1] + v.z * m.e[0][2] + v.w * m.e[0][3];
  result.y = v.x * m.e[1][0] + v.y * m.e[1][1] + v.z * m.e[1][2] + v.w * m.e[1][3];
  result.z = v.x * m.e[2][0] + v.y * m.e[2][1] + v.z * m.e[2][2] + v.w * m.e[2][3];
  result.w = v.x * m.e[3][0] + v.y * m.e[3][1] + v.z * m.e[3][2] + v.w * m.e[3][3];
  return result;
}

//~ Rect2

typedef struct Rect2 Rect2;
struct Rect2{
  union{
    struct{ V2 min, max; };
    struct{ f32 min_x, min_y, max_x, max_y; };
  };
};
static Rect2 rect2_from_center_and_extents(V2 center, V2 extents){
  Rect2 result = {0};
  result.min = v2_sub(center, extents);
  result.max = v2_add(center, extents);
  return result;
}
static Rect2 rect2_hull_of_2_points(V2 P0, V2 P1){
  Rect2 result = {0};
  result.min_x = MINIMUM(P0.x, P1.x);
  result.min_y = MINIMUM(P0.y, P1.y);
  result.max_x = MAXIMUM(P0.x, P1.x);
  result.max_y = MAXIMUM(P0.y, P1.y);
  return result;
}
static Rect2 rect2_hull_of_3_points(V2 P0, V2 P1, V2 P2){
  Rect2 result = {0};
  result.min_x = MINIMUM3(P0.x, P1.x, P2.x);
  result.min_y = MINIMUM3(P0.y, P1.y, P2.y);
  result.max_x = MAXIMUM3(P0.x, P1.x, P2.x);
  result.max_y = MAXIMUM3(P0.y, P1.y, P2.y);
  return result;
}

static V2 rect2_center(Rect2 rect){
  return v2_scalar_mul(v2_add(rect.max, rect.min), 0.5f);
}

static V2 map_normalized_onto_rect2(V2 P, Rect2 rect){
  V2 result = {0};
  result.x = (1 - P.x) * rect.min_x + P.x * rect.max_x;
  result.y = (1 - P.y) * rect.min_y + P.y * rect.max_y;
  return result;
}



static bool rect2_intersects(Rect2 a, Rect2 b){
  bool x = a.max_x >= b.min_x && b.max_x >= a.min_x;
  bool y = a.max_y >= b.min_y && b.max_y >= a.min_y;
  return x && y;
}
static Rect2 rect2_intersection(Rect2 a, Rect2 b){
  Rect2 result = {0};
  result.min_x = MAXIMUM(a.min_x, b.min_x);
  result.min_y = MAXIMUM(a.min_y, b.min_y);
  result.max_x = MINIMUM(a.max_x, b.max_x);
  result.max_y = MINIMUM(a.max_y, b.max_y);
  return result;
}
static bool is_v2_in_rect2(V2 P, Rect2 rect){
  return rect.min_x <= P.x && P.x <= rect.max_x
      && rect.min_y <= P.y && P.y <= rect.max_y;
}

static Rect2 rect2_translate(Rect2 rect, V2 v){
  return (Rect2){ v2_add(rect.min, v), v2_add(rect.max, v) };
}

static f32 rect2_dim_x(Rect2 rect){ return rect.max_x - rect.min_x; }
static f32 rect2_dim_y(Rect2 rect){ return rect.max_y - rect.min_y; }
static V2 rect2_dim(Rect2 rect){ return (V2){ .x = rect.max_x - rect.min_x, .y = rect.max_y - rect.min_y }; }

static Rect2 rect2_cut_left(Rect2 *r, f32 d){
  Rect2 result = *r;
  result.max_x = r->min_x + d;
  r->min_x += d;
  return result;
}
static Rect2 rect2_cut_right(Rect2 *r, f32 d){
  Rect2 result = *r;
  result.min_x = r->max_x - d;
  r->max_x -= d;
  return result;
}
static Rect2 rect2_cut_bottom(Rect2 *r, f32 d){
  Rect2 result = *r;
  result.max_y = r->min_y + d;
  r->min_y += d;
  return result;
}
static Rect2 rect2_cut_top(Rect2 *r, f32 d){
  Rect2 result = *r;
  result.min_y = r->max_y - d;
  r->max_y -= d;
  return result;
}

static Rect2 rect2_extend_left(Rect2 r, f32 d){
  Rect2 result = r;
  result.min_x -= d;
  return result;
}
static Rect2 rect2_extend_right(Rect2 r, f32 d){
  Rect2 result = r;
  result.max_x += d;
  return result;
}
static Rect2 rect2_extend_bottom(Rect2 r, f32 d){
  Rect2 result = r;
  result.min_y -= d;
  return result;
}
static Rect2 rect2_extend_top(Rect2 r, f32 d){
  Rect2 result = r;
  result.max_y += d;
  return result;
}

static Rect2 rect2_cut_margins(Rect2 r, f32 d){
  Rect2 result = r;
  result.min_x += d;
  result.max_x -= d;
  result.min_y += d;
  result.max_y -= d;
  return result;
}
static Rect2 rect2_cut_margins_xy(Rect2 r, V2 d){
  Rect2 result = r;
  result.min_x += d.x;
  result.max_x -= d.x;
  result.min_y += d.y;
  result.max_y -= d.y;
  return result;
}

// NOTE: This is for user interface purposes, so we prefer min_x and max_y being correct.
static Rect2 rect2_fit_other_rect_inside(Rect2 inner, Rect2 outer){
  Rect2 result = inner;
  if(inner.min_x < outer.min_x){
    f32 d = outer.min_x - inner.min_x;
    result.min_x += d;
    result.max_x += d;
  }else if(inner.max_x > outer.max_x){
    f32 d = outer.max_x - inner.max_x;
    result.min_x += d;
    result.max_x += d;
  }

  if(inner.max_y > outer.max_y){
    f32 d = outer.max_y - inner.max_y;
    result.min_y += d;
    result.max_y += d;
  }else if(inner.min_y < outer.min_y){
    f32 d = outer.min_y - inner.min_y;
    result.min_y += d;
    result.max_y += d;
  }
  return result;
}


typedef struct Rect2i Rect2i;
struct Rect2i{
  union{
    struct{ i32 min_x, min_y, max_x, max_y; };
  };
};


//~ Rect3

typedef struct Rect3 Rect3;
struct Rect3{
  V3 min, max;
};
static Rect3 rect3_from_center_and_extents(V3 center, V3 extents){
  Rect3 result = {0};
  result.min = v3_sub(center, extents);
  result.max = v3_add(center, extents);
  return result;
}
static V3 rect3_center(Rect3 rect){
  return (V3){ 0.5f * (rect.min.x + rect.max.x), 0.5f * (rect.min.y + rect.max.y), 0.5f * (rect.min.z + rect.max.z) };
}
static bool rect3_intersects(Rect3 a, Rect3 b){
  bool x = a.max.x >= b.min.x && b.max.x >= a.min.x;
  bool y = a.max.y >= b.min.y && b.max.y >= a.min.y;
  bool z = a.max.z >= b.min.z && b.max.z >= a.min.z;
  return x && y && z;
}
static V3 rect3_dim(Rect3 rect){
  V3 result = v3_sub(rect.max, rect.min);
  return result;
}

// Returns `true` if `outer` fully contains `inner`
static bool rect3_contains(Rect3 outer, Rect3 inner){
  bool result = false;
  if(1
    && outer.min.e[0] <= inner.min.e[0] && inner.max.e[0] <= outer.max.e[0]
    && outer.min.e[1] <= inner.min.e[1] && inner.max.e[1] <= outer.max.e[1]
    && outer.min.e[2] <= inner.min.e[2] && inner.max.e[2] <= outer.max.e[2])
  {
    result = true;
  }
  return result;
}

//~ Pseudorandom number generation

#include <sys/random.h>
static u64 get_entropy_from_os_u64(){
  u64 result;
  ssize_t getrandom_return = getrandom(&result, sizeof(result), 0);
  int getrandom_errno = errno;
  if(getrandom_return < 0){
    panic("getrandom failed: %s", strerror(getrandom_errno));
  }else if(getrandom_return != sizeof(result)){
    panic("getrandom did not return as many random bytes as we asked for");
  }
  return result;
}

//
// BEGIN PCG RANDOM
//
// This is a modified version of O'Neill's code which has the following license:
//
// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
//

typedef struct { uint64_t state;  uint64_t inc; } PCG_State;

static PCG_State pcg_create_with_os_entropy(){
  PCG_State result = {0};
  result.state = get_entropy_from_os_u64();
  result.inc = get_entropy_from_os_u64() | LIT_U64(1);
  return result;
}

static uint32_t pcg_random_u32(PCG_State* rng){
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static u64 pcg_random_u64(PCG_State *rng){
  // TODO: This is apparently not a good way of doing this, but I am too lazy too fix it for now.
  return (u64)pcg_random_u32(rng) | ((u64)pcg_random_u32(rng) << LIT_U64(32));
}

// NOTE(hanna): With my limited understanding of random number generation this should
// be a okay way of generating a random f32
static f32 pcg_random_f32_01(PCG_State *rng){
  u32 value = pcg_random_u32(rng);
  return (f32)((f64)value / (f64)0xffffffff);
}
static f32 pcg_random_f32(PCG_State *rng, f32 a, f32 b){
  return pcg_random_f32_01(rng) * (b - a) + a;
}

//
// END PCG RANDOM
//

// NOTE(hanna - 2021-05-14): I don't know much about PRNGs but here are some resources I have found:
// https://espadrine.github.io/blog/posts/a-primer-on-randomness.html
// https://burtleburtle.net/bob/rand/talksmall.html
//

//~ Dynamic arrays

typedef struct Array_any Array_any;
struct Array_any{
  void *e;
  u32 count;
  u32 capacity;
};
COMPILE_TIME_ASSERTION(sizeof(Array_any) == 16);

// TODO IMPORTANT: We should verify that the trick with Array_any does not break C aliasing rules

#define Array(_Type_) COMBINE2(Array_, _Type_)
#define DECLARE_ARRAY_TYPE(_Type_) \
  typedef union Array(_Type_){ \
    struct{ _Type_ *e; u32 count; u32 capacity; }; \
    Array_any as_any_array; \
  } Array(_Type_)

DECLARE_ARRAY_TYPE(bool);
DECLARE_ARRAY_TYPE(u8);
DECLARE_ARRAY_TYPE(u16);
DECLARE_ARRAY_TYPE(u32);
DECLARE_ARRAY_TYPE(u64);
DECLARE_ARRAY_TYPE(i8);
DECLARE_ARRAY_TYPE(i16);
DECLARE_ARRAY_TYPE(i32);
DECLARE_ARRAY_TYPE(i64);
DECLARE_ARRAY_TYPE(f32);
DECLARE_ARRAY_TYPE(f64);

DECLARE_ARRAY_TYPE(U8);
DECLARE_ARRAY_TYPE(U16);
DECLARE_ARRAY_TYPE(U32);
DECLARE_ARRAY_TYPE(U64);
DECLARE_ARRAY_TYPE(I8);
DECLARE_ARRAY_TYPE(I16);
DECLARE_ARRAY_TYPE(I32);
DECLARE_ARRAY_TYPE(I64);
DECLARE_ARRAY_TYPE(F32);
DECLARE_ARRAY_TYPE(F64);

DECLARE_ARRAY_TYPE(String);
DECLARE_ARRAY_TYPE(V2);
DECLARE_ARRAY_TYPE(V3);
DECLARE_ARRAY_TYPE(V4);

#define array_create(_Type_, _allocator_) ( (Array(_Type_)) { .e = (_Type_*)allocator_get_stub((_allocator_)) } )

static void _array_reserve(Array_any *array, u64 needed_capacity, size_t element_size, size_t element_align){
  assert(needed_capacity < UINT32_MAX);
  assert(element_size < UINT32_MAX);
  assert(array->e && "Before using a array you need to initialize it with array_create");

  if(array->capacity < needed_capacity){
    u64 old_size = array->capacity * (u64)element_size;
    size_t expand_size = needed_capacity * element_size;
    if(allocator_expand(array->e, old_size, expand_size)){
      array->capacity = needed_capacity;
    }else{
      u64 new_capacity = MAXIMUM(needed_capacity, array->capacity * 2);
      u64 realloc_size = new_capacity * (u64)element_size;
      allocator_realloc_noclear(&array->e, old_size, realloc_size, element_align);
      array->capacity = new_capacity;
      assert(array->e);
    }
  }
}
#define array_reserve(_array_, ...) \
  (_array_reserve(&(_array_)->as_any_array, (__VA_ARGS__), sizeof((_array_)->e[0]), __alignof__((_array_)->e[0])))

static void _array_destroy(Array_any *array, size_t element_size){
  size_t size = array->capacity * element_size;
  allocator_free(array->e, size);
  *array = (Array_any){0};
}

#define array_destroy(_array_) (_array_destroy(&(_array_)->as_any_array, sizeof((_array_)->e[0])), (_array_)->as_any_array = (Array_any){0})

static void _array_set_count_clear(Array_any *array, u64 new_count, size_t element_size, size_t element_align){
  _array_reserve(array, new_count, element_size, element_align);
  if(new_count > array->count){
    memset((u8*)array->e + element_size * array->count,
           0,
           (new_count - array->count) * element_size);
  }
  array->count = new_count;
}
#define array_set_count_clear(_array_, ...) \
  ( _array_set_count_clear(&(_array_)->as_any_array, (__VA_ARGS__), sizeof((_array_)->e[0]), __alignof__((_array_)->e[0])) )

static void _array_set_count_noclear(Array_any *array, u64 new_count, size_t element_size, size_t element_align){
  _array_reserve(array, new_count, element_size, element_align);
  array->count = new_count;
}
#define array_set_count_noclear(_array_, ...) \
  (_array_set_count_noclear(&(_array_)->as_any_array, (__VA_ARGS__), sizeof((_array_)->e[0]), __alignof__((_array_)->e[0])))

static i64 _array_get_element(Array_any *array, i64 index){
  assert(index >= 0);
  assert(index < array->count);
  i64 result = index;
  return result;
}
#define array_get_element(_array_, ...) \
  ((_array_)->e + _array_get_element((__VA_ARGS__)))

#define array_push(_array_, ...) \
  (_array_reserve(&(_array_)->as_any_array, (u64)(_array_)->count + 1, sizeof((_array_)->e[0]), __alignof__((_array_)->e[0])), \
   ((_array_)->e[(_array_)->count] = (__VA_ARGS__)), \
   ((_array_)->count += 1), \
   &((_array_)->e[(_array_)->count - 1]))

static void _array_insert(Array_any *array, u64 insert_index, size_t element_size, size_t element_align){
  assert(insert_index <= array->count);

  _array_reserve(array, array->count + 1, element_size, element_align);
  memmove((u8*)array->e + (insert_index + 1) * element_size,
          (u8*)array->e + (insert_index + 0) * element_size,
          element_size * (array->count - insert_index));
}
#define array_insert(_array_, _index_, ...) \
  (_array_insert(&(_array_)->as_any_array, (_index_), sizeof((_array_)->e[0]), __alignof__((_array_)->e[0])), \
   ((_array_)->e[(_index_)] = (__VA_ARGS__)), \
   ((_array_)->count += 1), /* NOTE: This allows for doing things like array_insert(&array, array.count, ...) */ \
   &((_array_)->e[(_index_)]))

static void _array_delete_range(Array_any *array, size_t element_size, u64 begin, u64 end){
  assert(begin <= end);
  assert(end <= array->count);

  memmove((u8*)array->e + begin * element_size,
          (u8*)array->e + end * element_size,
          element_size * (array->count - end));
  array->count -= (end - begin);
}
#define array_delete_at(_array_, _index_) \
  (_array_delete_range(&(_array_)->as_any_array, sizeof((_array_)->e[0]), (_index_), (_index_) + 1))

#define array_delete_range(_array_, _begin_, _end_) \
  (_array_delete_range(&(_array_)->as_any_array, sizeof((_array_)->e[0]), (_begin_), (_end_)))

#define array_delete_at_fast(_array_, _index_) \
  (assert((_array_)->count > 0), \
   (_array_)->e[(_index_)] = (_array_)->e[(_array_)->count - 1], \
   (_array_)->count -= 1)

#define array_pop(_array_) \
  (assert((_array_)->count > 0), (_array_)->e[(_array_)->count -= 1])

static Array_any _array_copy(Array_any *src, size_t element_size, size_t element_align){
  Array_any result = {0};
  Allocator *allocator = get_allocator(src->e);
  result.e = allocator_get_stub(allocator);
  _array_reserve(&result, src->count, element_size, element_align);
  memcpy(result.e, src->e, (size_t)element_size * (size_t)src->count);
  return result;
}
#define array_copy(_dst_, _src_) \
  ( (_dst_)->as_any_array = _array_copy(&(_src_)->as_any_array, sizeof((_src_)->e[0]), __alignof__((_src_)->e[0])) )

static String array_u8_as_string(Array(u8) array){
  return (String){ .data = array.e, .size = array.count };
}
static void array_u8_push_string(Array(u8) *array, String string){
  array_reserve(array, array->count + string.size);
  fiz(string.size){
    array_push(array, string.data[i]);
  }
}

//~ UTF32 Strings!

typedef struct StringUTF32 StringUTF32;
struct StringUTF32{
  u32 *data;
  size_t count;
};
CT_ASSERT(sizeof(StringUTF32) == 16);

static StringUTF32 string_utf32_from_array_u32(Array(u32) array){
  return (StringUTF32){ .data = array.e, .count = array.count };
}
static bool string_utf32_equals(StringUTF32 a, StringUTF32 b){
  return memory_equals(a.data, a.count * 4, b.data, b.count * 4);
}
static StringUTF32 string_utf32_from_utf8(Allocator *allocator, String string){
  Array(u32) result = array_create(u32, allocator);
  array_reserve(&result, string.size); // Expect most input to be ASCII. Worst case we allocate 4x the memory

  for(UTF8Iterator iter = iterate_utf8(string);
    iter.valid;
    advance_utf8_iterator(&iter))
  {
    array_push(&result, iter.codepoint);
  }

  return string_utf32_from_array_u32(result);
}

//
// FILE UTILITY
//

typedef struct EntireFile EntireFile;
struct EntireFile{
  bool ok;
  u8 *content;
  size_t size;
};
static EntireFile read_entire_file(String path, Allocator *allocator){
  EntireFile result = {0};

  OSFile file = os_open_file_input(path);
  if(file.value){
    size_t size = os_get_file_size(file);
    u8 *content = allocator_push_items_noclear(allocator, u8, size);
    if(os_read_from_file(file, 0, content, size)){
      result = (EntireFile){ .ok = true, .content = content, .size = size };
    }
  }
  os_close_file(file);

  return result;
}

static bool dump_string_to_file(String path, String string){
  bool result = false;

  OSFile file = os_open_file_output(path);
  if(file.value){
    bool error = false;
    os_write_to_file(file, 0, string.data, string.size, &error);
    result = !error;
  }
  os_close_file(file);

  return result;
}

static void string_to_lines(String text, Array(String) *out){
  i64 cursor = 0;
  while(cursor < text.size){
    i64 line_begin = cursor;
    while(cursor < text.size && text.data[cursor] != '\n'){
      cursor += 1;
    }
    i64 line_end = cursor;
    cursor += 1; // skip '\n'

    String line = substring(text, line_begin, line_end);
    array_push(out, line);
  }
}

//
// NOTE: String Builder
//

typedef struct SBChunk SBChunk;
struct SBChunk{
  SBChunk *next;
  u32 capacity;
  u32 cursor;
  u8 data[0];
};

typedef struct StringBuilder StringBuilder;
struct StringBuilder{
  Allocator *allocator;
  SBChunk *first_chunk;
  SBChunk *last_chunk;
  size_t total_size;
};

static StringBuilder sb_create(Allocator *allocator){
  StringBuilder result = {0};
  result.allocator = allocator;
  return result;
}

static String sb_to_string(StringBuilder *sb, Allocator *allocator){
  assert(sb->total_size < UINT32_MAX);

  u8 *data = allocator_push_items_noclear(allocator, u8, sb->total_size);
  size_t cursor = 0;
  for(SBChunk *chunk = sb->first_chunk; chunk; chunk = chunk->next){
    memcpy(data + cursor, chunk->data, chunk->cursor);
    cursor += chunk->cursor;
  }
  assert(cursor == sb->total_size);

  String result = { .data = data, .size = (u32)sb->total_size };
  return result;
}

static u8* sb_append_buffer(StringBuilder *sb, size_t bytes){
  u8 *result = NULL;
  SBChunk *chunk = sb->last_chunk;

  if(!chunk || chunk->cursor + bytes > chunk->capacity){
    u32 capacity = MAXIMUM(bytes, KILOBYTES(8));
    SBChunk *new_chunk = (SBChunk*)allocator_push_items_noclear(sb->allocator, u8, sizeof(SBChunk) + capacity);
    *new_chunk = (SBChunk){ .capacity = capacity };
    if(chunk){
      chunk->next = new_chunk;
      sb->last_chunk = new_chunk;
    }else{
      sb->first_chunk = sb->last_chunk = new_chunk;
    }
    chunk = new_chunk;
  }
  result = chunk->data + chunk->cursor;
  chunk->cursor += bytes;
  sb->total_size += bytes;

  return result;
}
static void sb_append_string(StringBuilder *sb, String string){
  u8 *data = sb_append_buffer(sb, string.size);
  memcpy(data, string.data, string.size);
}

static void sb_append_u8(StringBuilder *sb, u8 value){
  u8 *data = sb_append_buffer(sb, 1);
  *data = value;
}

static void sb_vprintf(StringBuilder *sb, const char *format, va_list arg_list1){
  va_list arg_list2;
  va_copy(arg_list2, arg_list1);

  int required_bytes = stbsp_vsnprintf(NULL, 0, format, arg_list1);

  // HACK: stbsp uses C strings so we must leave room for the zero terminator, but we don't want to keep it
  u8 *data = sb_append_buffer(sb, required_bytes + 1);
  assert(sb->last_chunk->cursor > 0);
  sb->last_chunk->cursor -= 1; // we don't want the zero terminator to take space here
  sb->total_size -= 1; // no zero terminator
  int required_bytes2 = stbsp_vsnprintf((char*)data, required_bytes + 1, format, arg_list2);

  assert(required_bytes == required_bytes2);

  va_end(arg_list2);
}

static void sb_printf(StringBuilder *sb, const char *format, ...){
  va_list list;
  va_start(list, format);
  sb_vprintf(sb, format, list);
  va_end(list);
}

static void sb_append_sb(StringBuilder *out, StringBuilder *in){
  // TODO: Could be optimized to utilize the current chunk fully before using the next one
  u8 *at = sb_append_buffer(out, in->total_size);
  for(SBChunk *chunk = in->first_chunk; chunk; chunk = chunk->next){
    memcpy(at, chunk->data, chunk->cursor);
    at += chunk->cursor;
  }
}

//
// Error storage
//

typedef struct Error Error;
struct Error{
  Error *next;

  const char *file;
  const char *procedure_signature;
  int line;

  String string;
};

typedef struct Errors Errors;
struct Errors{
  Allocator *allocator;
  Error *first, *last;
};
static Errors errors_create(Allocator *allocator){
  Errors result = {0};
  result.allocator = allocator;
  return result;
}
static void errors_vpushf(Errors *errors, const char *file, const char *procedure_signature, int line, const char *format, va_list list){
  Error *error = allocator_push_item_clear(errors->allocator, Error);
  if(errors->first){
    assert(errors->last);
    errors->last->next = error;
    errors->last = error;
  }else{
    assert(!errors->last);
    errors->first = errors->last = error;
  }
  error->file = file;
  error->procedure_signature = procedure_signature;
  error->line = line;
  error->string = allocator_push_vprintf(errors->allocator, format, list);
}
static void _errors_pushf(Errors *errors, const char *file, const char *procedure_signature, int line, const char *format, ...){
  va_list list;
  va_start(list, format);
  errors_vpushf(errors, file, procedure_signature, line, format, list);
  va_end(list);
}
#define errors_pushf(_errors_, ...) _errors_pushf((_errors_), __FILE__, __PRETTY_FUNCTION__, __LINE__, __VA_ARGS__)
static bool errors_any(Errors *errors){
  return (errors->first != NULL);
}
static String errors_to_string(Errors *errors, Allocator *allocator){
  StringBuilder sb = sb_create(allocator); // TODO: Proper error
  for(Error *error = errors->first; error; error = error->next){
    sb_printf(&sb, "{%s:%d:%s: %.*s}", error->file, error->line, error->procedure_signature, StrFormatArg(error->string));
  }
  return sb_to_string(&sb, allocator);
}

//
// Stream
//

typedef struct StreamChunk StreamChunk;
struct StreamChunk{
  StreamChunk *next;
  size_t size;
  u8 data[0];
};

typedef struct Stream Stream;
struct Stream{ // TODO: Consider using a cyclic buffer for this instead.
  Allocator *allocator;

  size_t total_bytes;
  size_t cursor;
  StreamChunk *current_chunk;
  StreamChunk *last_chunk;
};

static Stream stream_create(Allocator *allocator){
  Stream result = {0};
  result.allocator = allocator;
  return result;
}
static void stream_destroy(Stream *stream){
  for(StreamChunk *chunk = stream->current_chunk; chunk;){
    StreamChunk *next = chunk->next;
    allocator_free(chunk, sizeof(StreamChunk) + chunk->size);
    chunk = next;
  }
  *stream = (Stream){0};
}

static void stream_feed(Stream *stream, u8 *data, size_t size){
  StreamChunk *chunk = (StreamChunk*)allocator_alloc_noclear(stream->allocator, sizeof(StreamChunk) + size, 1);
  *chunk = (StreamChunk){0};
  chunk->size = size;
  memcpy(chunk->data, data, size);
  stream->total_bytes += size;

  if(stream->current_chunk){
    stream->last_chunk->next = chunk;
    stream->last_chunk = chunk;
  }else{
    stream->current_chunk = stream->last_chunk = chunk;
  }
}
static void stream_feed_u8(Stream *stream, u8 value)  { stream_feed(stream, (u8*)&value, 1); }
static void stream_feed_u16(Stream *stream, u16 value){ stream_feed(stream, (u8*)&value, 2); }
static void stream_feed_u32(Stream *stream, u32 value){ stream_feed(stream, (u8*)&value, 4); }
static void stream_feed_u64(Stream *stream, u64 value){ stream_feed(stream, (u8*)&value, 8); }

static void stream_consume_chunk(Stream *stream){
  assert(stream->current_chunk);

  StreamChunk *next = stream->current_chunk->next;
  allocator_free(stream->current_chunk, sizeof(StreamChunk) + stream->current_chunk->size);
  stream->current_chunk = next;
  if(!next) stream->last_chunk = NULL;
}

static bool stream_consume(Stream *stream, u8 *out, size_t out_size){
  bool result = false;

  size_t out_cursor = 0;
  size_t chunk_cursor = stream->cursor;
  size_t total_bytes = stream->total_bytes;
  StreamChunk *chunk = stream->current_chunk;
  for(; chunk; (chunk = chunk->next), (chunk_cursor = 0)){
    size_t size = MINIMUM(chunk->size - chunk_cursor, out_size - out_cursor);
    if(out){
      memcpy(out + out_cursor, chunk->data + chunk_cursor, size);
    }
    out_cursor += size;
    chunk_cursor += size;
    total_bytes -= size;

    if(out_cursor >= out_size) break;
  }
  if(out_cursor == out_size){
    result = true;
    for(; stream->current_chunk != chunk;){
      stream_consume_chunk(stream);
    }

    stream->total_bytes = total_bytes;
    stream->cursor = chunk_cursor;
  }

  return result;
}

static bool stream_consume_line_crlf(Stream *stream, Allocator *allocator, String *_line){
  bool result = false;
  String line = {0};

  size_t size = 0;
  size_t cursor = stream->cursor;
  for(StreamChunk *chunk = stream->current_chunk; chunk; (chunk = chunk->next), (cursor = 0)){
    for(; cursor < chunk->size; cursor += 1, size += 1){
      if(cursor + 2 <= chunk->size && chunk->data[cursor + 0] == '\r' && chunk->data[cursor + 1] == '\n'){
        goto found;
      }
    }
  }
  if(0){
    found:;
    result = true;
    assert(size < UINT32_MAX);
    line = (String){ .data = allocator_push_items_noclear(allocator, u8, size), .size = (u32)size };
    bool status = stream_consume(stream, line.data, line.size);
    assert(status);
    status = stream_consume(stream, NULL, 2);
    assert(status);
  }

  *_line = line;
  return result;
}

//
// Endianess conversion
//

static u64 u64_swap_endianess(u64 value){
  u8 *bytes = (u8*)&value;
  u8_swap(&bytes[0], &bytes[7]);
  u8_swap(&bytes[1], &bytes[6]);
  u8_swap(&bytes[2], &bytes[5]);
  u8_swap(&bytes[3], &bytes[4]);
  return *(u64*)bytes;
}
static u16 u16_swap_endianess(u16 value){
  u8 *bytes = (u8*)&value;
  u8_swap(&bytes[0], &bytes[1]);
  return *(u16*)bytes;
}

#endif // HK_UTIL_H

//
// NOTE: LICENSE
//

/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
*/

