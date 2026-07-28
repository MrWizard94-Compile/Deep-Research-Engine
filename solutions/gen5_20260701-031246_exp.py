import numpy as np

def svd_centrality(matrix):
    U, s, Vt = np.linalg.svd(matrix)
    centrality_scores = np.sum(np.abs(Vt), axis=1)
    return centrality_scores / np.sum(centrality_scores)

def simulate_cache_replacement(num_items, num_accesses, cache_size):
    access_matrix = np.random.randint(0, 2, size=(num_items, num_accesses))
    centrality_scores = svd_centrality(access_matrix)

    cache = np.argsort(centrality_scores)[-cache_size:]
    hit_count = 0

    for i in range(num_accesses):
        if access_matrix[cache, i].any():
            hit_count += 1

    return hit_count / num_accesses

if __name__ == "__main__":
    num_items = 100
    num_accesses = 1000
    cache_size = 10

    cache_hit_rate = simulate_cache_replacement(num_items, num_accesses, cache_size)
    print(f"METRIC cache_hit_rate={cache_hit_rate}")