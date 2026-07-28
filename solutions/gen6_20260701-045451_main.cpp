#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <array>

template<typename T>
class SegmentedQueue {
public:
    struct Segment {
        std::atomic<int> head;
        std::atomic<int> tail;
        std::array<T, 1024> buffer;

        Segment() : head(0), tail(0) {}
    };

    SegmentedQueue(int num_segments) : segments(num_segments) {}

    void enqueue(T value) {
        int segment_index = (segments.size() - 1) & ((head.load(std::memory_order_relaxed) / 1024));
        Segment& segment = segments[segment_index];
        while (true) {
            int tail = segment.tail.load(std::memory_order_acquire);
            if (tail < (segment.head.load(std::memory_order_relaxed) + 1024)) {
                segment.buffer[tail % 1024] = value;
                segment.tail.store(tail + 1, std::memory_order_release);
                break;
            }
        }
        head.fetch_add(1, std::memory_order_relaxed);
    }

private:
    std::vector<Segment> segments;
    std::atomic<int> head{0};
};

void producer(SegmentedQueue<int>& queue, int id, int num_items) {
    for (int i = 0; i < num_items; ++i) {
        queue.enqueue(id * num_items + i);
    }
}

int main() {
    const int num_producers = 4;
    const int items_per_producer = 1000;
    const int num_segments = 8;

    SegmentedQueue<int> queue(num_segments);

    std::vector<std::thread> producers;
    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back(producer, std::ref(queue), i, items_per_producer);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (auto& p : producers) {
        p.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double throughput_per_producer = (static_cast<double>(num_producers * items_per_producer) /
                                      std::chrono::duration<double>(end - start).count()) * 1000.0;

    std::cout << "METRIC throughput_per_producer=" << throughput_per_producer << std::endl;
    return 0;
}