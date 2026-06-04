# tile-cuda Makefile
# CUDA toolkit detection, multi-arch compilation

# Try to auto-detect CUDA toolkit
CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc

# Source files
SRCS      := $(wildcard src/*.cu)
BENCH_SRC := benches/bench_cuda.cu
TEST_SRC  := tests/test_cuda.cu

# Common flags
NVCC_FLAGS := -std=c++17 -O3 \
              -Iinclude \
              -Xcompiler -Wall,-Wextra

# Multi-arch: sm_75 (Turing), sm_80 (Ampere), sm_89 (Ada), sm_90 (Hopper)
ARCH_FLAGS := -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_90,code=sm_90

# Output
BUILD_DIR := build
LIB       := $(BUILD_DIR)/libtile_cuda.a

# Targets
.PHONY: all test bench clean

all: $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build static library from all kernel objects
$(LIB): $(SRCS:.cu=.o) | $(BUILD_DIR)
	$(AR) rcs $@ $^

# Compile .cu → .o
%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -c $< -o $@

# Test binary
test: test_cuda
	./test_cuda

test_cuda: $(TEST_SRC) $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(SRCS) $(TEST_SRC) -o $(BUILD_DIR)/test_cuda

# Benchmark binary
bench: bench_cuda
	./bench_cuda

bench_cuda: $(BENCH_SRC) $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(SRCS) $(BENCH_SRC) -o $(BUILD_DIR)/bench_cuda

clean:
	rm -rf $(BUILD_DIR)
	rm -f src/*.o

# Check if nvcc exists
nvcc-check:
	@which $(NVCC) 2>/dev/null && echo "nvcc found" || echo "nvcc not found at $(NVCC)"
