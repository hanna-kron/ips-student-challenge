Hi!

To compile and also do profiling runs of the project, run ./build.sh
As I do all my development on Linux I don't have any support for Windows at the moment.

These are the relevant source files:

main.c              --- contains most of the code for this project.
common.h            --- configuration options, also code shared by main.c and kernel.cu
kernel.cu           --- the CUDA code for this project
gen-positions.c     ---
../shared/hk_util.h --- my utility code I use in all my projects
../shared/qsort.inl --- sorting in C without function pointers.
../3rdparty/        --- 3rdparty code used in the project


/Hanna

