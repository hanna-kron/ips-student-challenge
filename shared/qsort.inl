/*
qsort.inl - quick-sort template - qsort without function pointers.
Created: 2022-03-26
Author: hanna


MODIFIED VERSION OF qsort.inl FROM SYMBOLS PROJECT


You need the following defines:

#define QSORT_NAME
Name of the sorting procedure.

#define QSORT_TYPE
Data type you are sorting.

#define QSORT_COMPARE(_userdata_, _a_, _b_)
Returns true if (_a_) < (_b_) for descending sort. Returns (_a_) > (_b_) for ascending sort.
*/

//#include "qtcreator_is_garbage.h"
#ifdef YOU_ARE_QTCREATOR_NOT_THE_COMPILER
#define QSORT_NAME sort_the_integers
#define QSORT_TYPE int
#define QSORT_COMPARE(_userdata_, _a_, _b_) 0
#endif

#ifndef QSORT_NAME
#error
#endif
#ifndef QSORT_TYPE
#error
#endif
#ifndef QSORT_COMPARE
#error
#endif

// TODO: Proper temporary memory!
static void QSORT_NAME(void *userdata, QSORT_TYPE *values, size_t count, Allocator *allocator){
  if(count > 0){
    //mTemp *temp = m_begin_temp();
    assert(count < UINT32_MAX);

    Array(u32) stack = array_create(u32, allocator); // TODO: Proper temporary memory!!!

    u32 left = 0;
    u32 right = count - 1;

    while(true){
      assert(left <= right);
      assert(left < count);
      assert(right < count);
      u32 P = left;
      QSORT_TYPE pivot = values[P];

      // Move pivot to the right place
      for(u32 Q = left + 1; // +1 to skip pivot
          Q <= right;
          Q += 1)
      {
        if(QSORT_COMPARE(userdata, pivot, values[Q])){
          // Q -> P
          // [ P + 1, Q - 1 ] -> [ P + 2, Q ]
          values[P] = values[Q];
          memmove(values + P + 2, values + P + 1, sizeof(QSORT_TYPE) * (Q - P - 1));
          P += 1;
        }
      }

      values[P] = pivot;
      retry:;
      if(left + 1 < P){ // Go further down left branch
        array_push(&stack, right); // Remember the right branch [ P + 1, right ]
        // Go down left branch [ left, P - 1 ]
        left = left;
        right = P - 1;
      }else if(P + 1 < right){ // Go directly to right branch [ P + 1, right ], if any
        left = P + 1;
        right = right;
      }else if(stack.count > 0){ // Otherwise recall a right side stored in the stack, as everything left of the current `right` cursor is sorted.
        // Recall the right branch [ P + 1, right ], based on [ left, P - 1 ] + the stored value
        left = right + 2;
        right = array_pop(&stack);

        if(left > right){ // This is when the right branch [ P + 1, right ] had P = right, i.e. it is an empty interval
          goto retry;
        }
      }else{
        break;
      }
    }

    //m_end_temp(temp);
    array_destroy(&stack);
  }
}

#undef QSORT_NAME
#undef QSORT_TYPE
#undef QSORT_COMPARE
