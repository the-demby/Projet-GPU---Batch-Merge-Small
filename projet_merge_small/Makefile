# Makefile
NVCC = nvcc
INCLUDES = -Iinclude
KERNELS = kernels/merge_kernels.cu

# Compilation des exécutables
mergeSmall: src/test_mergeSmall.cu $(KERNELS)
	$(NVCC) $(INCLUDES) $^ -o mergeSmall

batchMerge: src/test_batchMerge.cu $(KERNELS)
	$(NVCC) $(INCLUDES) $^ -o batchMerge

# Exécution
run_small: mergeSmall
	./mergeSmall

run_batch: batchMerge
	./batchMerge

clean:
	rm -f mergeSmall batchMerge

benchmark: src/benchmark_batchMerge.cu $(KERNELS)
	$(NVCC) $(INCLUDES) $^ -o benchmark

run_bench: benchmark
	./benchmark
benchmark_shared: src/benchmark_batchMerge_shared.cu $(KERNELS)
	$(NVCC) $(INCLUDES) $^ -o benchmark_shared

run_bench_shared: benchmark_shared
	./benchmark_shared
benchmark_one: src/benchmark_onePerBlock.cu $(KERNELS)
	$(NVCC) $(INCLUDES) $^ -o benchmark_one

run_bench_one: benchmark_one
	./benchmark_one
benchmark_coop: src/benchmark_coopWarp.cu $(KERNELS)
	$(NVCC) $(INCLUDES) $^ -o benchmark_coop

run_bench_coop: benchmark_coop
	./benchmark_coop
