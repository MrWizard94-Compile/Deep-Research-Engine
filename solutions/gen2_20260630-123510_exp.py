import numpy as np
import time

class HashingIndex:
    def __init__(self, dim, num_buckets):
        self.dim = dim
        self.num_buckets = num_buckets
        self.hash_table = [[] for _ in range(num_buckets)]

    def hash_function(self, vector):
        return int(sum(vector) % self.num_buckets)

    def insert(self, vector, index):
        bucket_id = self.hash_function(vector)
        self.hash_table[bucket_id].append((vector, index))

    def query(self, query_vector, k=1):
        bucket_id = self.hash_function(query_vector)
        candidates = self.hash_table[bucket_id]

        distances = []
        for vector, idx in candidates:
            distance = np.linalg.norm(vector - query_vector)
            distances.append((distance, idx))

        return sorted(distances)[:k]

def generate_sparse_vectors(num_vectors, dim, density):
    vectors = []
    for _ in range(num_vectors):
        vector = np.zeros(dim)
        non_zero_indices = np.random.choice(dim, int(density * dim), replace=False)
        vector[non_zero_indices] = np.random.rand(len(non_zero_indices))
        vectors.append(vector)
    return vectors

def main():
    num_vectors = 1000
    dim = 1000
    density = 0.01
    num_buckets = 1000

    vectors = generate_sparse_vectors(num_vectors, dim, density)

    index = HashingIndex(dim, num_buckets)
    for i, vector in enumerate(vectors):
        index.insert(vector, i)

    query_vector = np.random.rand(dim)
    start_time = time.time()
    nearest_neighbors = index.query(query_vector, k=10)
    end_time = time.time()

    throughput = num_vectors / (end_time - start_time)
    print(f"METRIC throughput={throughput}")

if __name__ == "__main__":
    main()