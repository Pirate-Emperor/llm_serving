# Disaggregated LLM Serving Engine

## Overview

The **Disaggregated LLM Serving Engine** is a "fullstack" training and inference architecture designed to multiplex compute-bound prompt processing with memory-bound decode steps. It allows you to train a LLaMA-based LLM architecture in PyTorch, then execute inference using highly optimized, zero-dependency C runtimes and bare-metal CUDA HGEMM kernels.

By leveraging chunked prefill scheduling and dynamic memory paging, very small LLMs can have surprisingly strong performance if you make the domain narrow enough. The engine supports models up to 7B/13B parameters in fp32 and quantized int8, executing entirely locally with minimal overhead.

---

## Part I: The C Inference Runtime

### Quick Start & Execution

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone [https://github.com/your-org/Disaggregated-LLM-Serving-Engine.git](https://github.com/your-org/Disaggregated-LLM-Serving-Engine.git)
cd Disaggregated-LLM-Serving-Engine

```

Now, let's run a baby LLaMA model in C. You need a model checkpoint. Download the 15M parameter model trained on the TinyStories dataset (~60MB download):

```bash
wget [https://huggingface.co/models/tinyllamas/resolve/main/stories15M.bin](https://huggingface.co/models/tinyllamas/resolve/main/stories15M.bin)

```

Compile and run the C code:

```bash
make run
./run stories15M.bin

```

You'll see the text stream a sample. On a standard M1/M2 chip, this runs at ~110 tokens/s. See the performance section for compile flags that can significantly speed this up. We can also try a slightly larger 42M parameter model:

```bash
wget [https://huggingface.co/models/tinyllamas/resolve/main/stories42M.bin](https://huggingface.co/models/tinyllamas/resolve/main/stories42M.bin)
./run stories42M.bin

```

You can also prompt the model with a prefix or a number of additional command line arguments, e.g. to sample at temperature 0.8 for 256 steps and with a prompt:

```bash
./run stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"

```

Quick note on sampling, the recommendation for ~best results is to sample with `-t 1.0 -p 0.9`, i.e. temperature 1.0 (default) but also top-p sampling at 0.9 (default). Intuitively, top-p ensures that tokens with tiny probabilities do not get sampled, so we can't get "unlucky" during sampling, and we are less likely to go "off the rails" afterwards. More generally, to control the diversity of samples use either the temperature (i.e. vary `-t` between 0 and 1 and keep top-p off with `-p 0`) or the top-p value (i.e. vary `-p` between 0 and 1 and keep `-t 1`), but not both.

### Standard LLaMA Models

As the neural net architecture is identical, we can also inference standard open-weight LLaMA models. To do this, we have to convert them into the internal binary format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for a 7B model:

```bash
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B

```

The export will take ~10 minutes or so and generate a 26GB file (the weights of the 7B model in float32) called `llama2_7b.bin` in the current directory. Once the export is done, we can run it:

```bash
./run llama2_7b.bin

```

You can also chat with the Chat models. Export the chat model exactly as above:

```bash
python export.py llama2_7b_chat.bin --meta-llama /path/to/7B-chat
./run llama2_7b_chat.bin -m chat

```

### INT8 Quantization

The default script uses a float32 forward pass, where the entire calculation of the forward pass is kept in fp32. This is very easy to understand as far as reference code goes, but it has the following downsides: the model checkpoint files are very large (it takes 4 bytes per every individual weight), and the forward pass is relatively slow.

The inference optimization employed in this engine is to quantize the model parameters to lower precision, giving up a little bit of correctness in return for smaller checkpoint sizes and faster forward passes (as most of the inference uses integer arithmetic). Only the weights that participate in matmuls are quantized. All the other parameters (e.g. especially the scale and bias in RMSNorm) are kept in float32, because these layers are very sensitive.

We additionally quantize the activations in the forward pass. This requires us to dynamically quantize and dequantize between float32 and int8 at runtime, which adds overhead. But the benefit is that now the majority of the calculations (the matmuls especially!) are using pure integer arithmetic, where both weights and activations enter as int8. This is where the speedups fundamentally come from. The version we use is the "Q8_0" quantization, where the 0 means that the weight quantization is symmetric around 0, quantizing to the range [-127, 127].

To export an int8 quantized model:

```bash
python export.py llama2_7b_q80.bin --version 2 --meta-llama path/to/llama/model/7B

```

This runs for a few minutes, but now creates only a 6.7GB file. Now let's inference them.

```bash
make runomp
OMP_NUM_THREADS=64 ./run llama2_7b.bin -n 40
OMP_NUM_THREADS=64 ./runq llama2_7b_q80.bin -n 40

```

This achieves a 3X speedup while reducing the checkpoint size by 4X.

---

## Part II: PyTorch Training & Custom Tokenizers

### Custom Tokenizers

In everything above, we've assumed the custom tokenizer with 32,000 tokens. However, in many boutique LLMs, using vocabulary this big might be overkill. If you have a small application you have in mind, you might be much better off training your own tokenizers. With smaller vocabs your model has fewer parameters (because the token embedding table is a lot smaller), the inference is faster (because there are fewer tokens to predict), and your average sequence length per example could also get smaller.

To train an example 4096-token tokenizer:

```bash
python dataset.py download
python dataset.py train_vocab --vocab_size=4096
python dataset.py pretokenize --vocab_size=4096

```

The `train_vocab` stage will call the `sentencepiece` library to train the tokenizer, storing it in a new file `data/tok4096.model`. This uses the Byte Pair Encoding algorithm that starts out with raw utf8 byte sequences of the text data and then iteratively merges the most common consecutive pairs of tokens to form the vocabulary.

When training your model, make sure to pass in the custom vocab size:

```bash
python train.py --vocab_source=custom --vocab_size=4096

```

Finally we are ready to run inference. We have to export our tokenizer in the `.bin` format:

```bash
python tokenizer.py --tokenizer-model=data/tok4096.model
./run out/model.bin -z data/tok4096.bin

```

### Training Guide

See the `train.py` script for more exotic launches and hyperparameter overrides. Set the max context length however you wish, depending on the problem: this should be the max number of tokens that matter to predict the next token. You want the *total* batch size per update to be somewhere around 100K tokens for medium-sized applications. You get there by first maxing out the batch_size to whatever your system allows, and then you want to increase `gradient_accumulation_steps` to be as high as necessary.

Finally, tune your learning_rate (LR). You want this to be as high as your training allows. Very small networks can get away with a large LR (e.g. 1e-3 or even higher). Large networks need lower LRs.

---

## Part III: Hardware Acceleration & Performance

### C CPU Optimizations (OpenMP)

There are many ways to potentially speed up this code depending on your system. The `make run` command currently uses the `-O3` optimization by default.
To get a much better performance, try to compile with `make runfast`. This turns on the `-Ofast` flag, which includes additional optimizations that may break compliance with the C/IEEE specifications, in addition to `-O3`.

Try `-march=native` to compile the program to use the architecture of the machine you're compiling on rather than a more generic CPU. This may enable additional optimizations and hardware-specific tuning such as improved vector instructions/width.

Big improvements can also be achieved by compiling with OpenMP, which "activates" the `#pragma omp parallel for` inside the matmul and attention, allowing the work in the loops to be split up over multiple processors.

```bash
clang -Ofast -fopenmp -march=native run.c -lm -o run
OMP_NUM_THREADS=4 ./run out/model.bin

```

### CUDA HGEMM (Half-Precision General Matrix Multiplication)

For GPU offloading, the engine implements several optimization methods of half-precision general matrix multiplication (HGEMM) using tensor core with WMMA API and MMA PTX instruction. The calculation expression is as follows, where the precision of matrix A (M * K), B (K * N) and C (M * N) is FP16. Through exploring various matrix tiling and optimization methods, the current performance between 256 to 16384 dimensions is not less than 95% of the performance of cublas, and in many scenarios, it exceeds the performance of cublas.

```text
C (M * N) = A (M * K) * B (K * N)

```

**Optimization Methods Deployed:**

* **Tiling:** 256 * 128 for block tiling size and 64 * 64 for warp tiling size
* **Coalescing Access:** using wide instruction access to global memory
* **Data Reuse:** using shared memory to reuse data of matrix A and B
* **Async Copy:** using asynchronous copy operation with non-blocking instruction
* **Bank Conflict:** using padding method for WMMA API and permuted method for MMA PTX instruction to eliminate bank conflict
* **L2 Cache:** using swizzle access mode to increase L2 cache hit ratio
* **Register Reuse:** calculating as "Right Left Right Left" for the internal tile of warp
* **Pg2s:** double-buffer algorithm using prefetching global memory to shared memory
* **Ps2r:** double-buffer algorithm using prefetching shared memory to register
* **Stage:** multi-buffer algorithm using prefetching global memory to shared memory

### Compile CUDA Kernels

**Environment:**

* OS: Linux
* Cmake Version: >= 3.12
* GCC Version: >= 4.8
* CUDA Version: >= 11.0

```bash
sudo apt-get install libgflags-dev ccache

```

**Build for NVIDIA A100 (Ampere):**

```bash
cd cuda_hgemm
./build.sh -a 80 -t Release -b OFF
./build.sh -a 80 -t Debug -b OFF

```

**Build for RTX3080Ti / RTX3090 / RTX A6000 (Ampere):**

```bash
cd cuda_hgemm
./build.sh -a 86 -t Release -b OFF
./build.sh -a 86 -t Debug -b OFF

```

### Run Performance Sample

```bash
./run_sample.sh

```

To process the data in the log and plot it as a line chart:

```bash
cd tools/performance
./performance.sh

```

The resulting architecture dynamically slices prompt prefill compute, and captures execution subgraphs to break through the memory-bandwidth wall during autoregressive decoding, significantly reducing Inter-Token Latency (ITL) on batched requests.

## License

This project is licensed under the Pirate-Emperor License. See the [LICENSE](LICENSE) file for details.

## Author

**Pirate-Emperor**

[![Twitter](https://skillicons.dev/icons?i=twitter)](https://twitter.com/PirateKingRahul)
[![Discord](https://skillicons.dev/icons?i=discord)](https://discord.com/users/1200728704981143634)
[![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/piratekingrahul)

[![Reddit](https://img.shields.io/badge/Reddit-FF5700?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/u/PirateKingRahul)
[![Medium](https://img.shields.io/badge/Medium-42404E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@piratekingrahul)

- GitHub: [Pirate-Emperor](https://github.com/Pirate-Emperor)
- Reddit: [PirateKingRahul](https://www.reddit.com/u/PirateKingRahul/)
- Twitter: [PirateKingRahul](https://twitter.com/PirateKingRahul)
- Discord: [PirateKingRahul](https://discord.com/users/1200728704981143634)
- LinkedIn: [PirateKingRahul](https://www.linkedin.com/in/piratekingrahul)
- Skype: [Join Skype](https://join.skype.com/invite/yfjOJG3wv9Ki)
- Medium: [PirateKingRahul](https://medium.com/@piratekingrahul)

Thank you for visiting this project!

---