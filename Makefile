.PHONY: test dlse-test dlse-bench dlse-experiment dlse-cuda clean

CUDA_ARCH ?= 86

test: dlse-test

dlse-test:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel
	ctest --test-dir build --output-on-failure

dlse-bench: dlse-test
	./build/bin/dlse_runtime_bench

dlse-experiment: dlse-test
	./build/bin/dlse_runtime_experiments dlse_runtime/results/experiments

dlse-cuda:
	cmake -S . -B build-cuda \
		-DDLSE_ENABLE_CUDA=ON \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	cmake --build build-cuda --parallel
	./build-cuda/bin/dlse_paged_attention_bench
	./build-cuda/bin/dlse_cuda_graph_bench 10000

clean:
	rm -rf build build-cuda
