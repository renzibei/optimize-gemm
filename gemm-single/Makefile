# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = gcc 
# OPT = -no-multibyte-chars
FLAGS = -O3
# -fopt-info
LDLIBS = -lpthread -lm -llapack -lopenblas -lrt -lcblas -I/usr/include/openblas -lgptl -lpapi -O
targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark-test.o benchmark.o gemm-naive.o gemm-blocked.o gemm-blas.o


.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-test : benchmark-test.o gemm-blocked.o
	$(CC) -o $@ $^ $(FLAGS) $(LDLIBS)

benchmark-naive : benchmark.o gemm-naive.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS)
benchmark-blocked : benchmark.o gemm-blocked.o
	$(CC) -o $@ $^ $(FLAGS) $(LDLIBS)
benchmark-blas : benchmark.o gemm-blas.o
	$(CC) -o $@ $^ $(FLAGS) $(LDLIBS)
%.o : %.c
	$(CC) -c $(CFLAGS) $(FLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
