import numpy as np
import time

def binary_quantize(matrix):
    return (matrix > 0).astype(np.int8)

def main():
    # Generate a random matrix of floats
    size = 1024
    matrix = np.random.rand(size, size)

    # Measure the execution time of binary quantization
    start_time = time.time()
    quantized_matrix = binary_quantize(matrix)
    end_time = time.time()

    # Calculate the execution time in seconds
    execution_time = end_time - start_time

    # Print the required metric line
    print(f"METRIC execution_time={execution_time}")

if __name__ == "__main__":
    main()