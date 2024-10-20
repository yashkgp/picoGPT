import time
from gpt2 import main as gpt2_main
from utils import load_encoder_hparams_and_params

def benchmark_generation(prompt, n_tokens_to_generate, model_size, use_speculative):
    start_time = time.time()
    gpt2_main(prompt, n_tokens_to_generate, model_size, use_generate_speculative=use_speculative)
    end_time = time.time()
    return end_time - start_time

def run_benchmark(prompt, n_tokens_to_generate, model_sizes):
    results = {}

    for model_size in model_sizes:
        print(f"Benchmarking {model_size} model...")
        
        # Warm-up run
        benchmark_generation(prompt, n_tokens_to_generate, model_size, False)
        benchmark_generation(prompt, n_tokens_to_generate, model_size, True)

        # Actual benchmark
        standard_time = benchmark_generation(prompt, n_tokens_to_generate, model_size, False)
        speculative_time = benchmark_generation(prompt, n_tokens_to_generate, model_size, True)

        improvement = (standard_time - speculative_time) / standard_time * 100
        results[model_size] = {
            "standard_time": standard_time,
            "speculative_time": speculative_time,
            "improvement": improvement
        }

    return results

def main():
    prompt = "In a world where artificial intelligence has become ubiquitous"
    n_tokens_to_generate = 50
    model_sizes = ["124M", "355M"]

    results = run_benchmark(prompt, n_tokens_to_generate, model_sizes)

    print("\nBenchmark Results:")
    print("==================")
    for model_size, data in results.items():
        print(f"\nModel Size: {model_size}")
        print(f"Standard Generation Time: {data['standard_time']:.4f} seconds")
        print(f"Speculative Generation Time: {data['speculative_time']:.4f} seconds")
        print(f"Improvement: {data['improvement']:.2f}%")

if __name__ == "__main__":
    main()
