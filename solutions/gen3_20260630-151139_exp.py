import numpy as np

# Define a simple task class
class Task:
    def __init__(self, execution_time, power_consumption):
        self.execution_time = execution_time
        self.power_consumption = power_consumption

# Function to calculate critical path execution time
def critical_path(tasks):
    return max(task.execution_time for task in tasks)

# Function to simulate scheduling based on critical path and power consumption
def schedule_tasks(tasks, power_budget):
    # Sort tasks by their execution time (critical path)
    sorted_tasks = sorted(tasks, key=lambda x: x.execution_time, reverse=True)

    total_execution_time = 0
    current_power_consumption = 0

    for task in sorted_tasks:
        if current_power_consumption + task.power_consumption <= power_budget:
            total_execution_time += task.execution_time
            current_power_consumption += task.power_consumption

    return total_execution_time

# Generate some sample tasks
np.random.seed(42)
num_tasks = 10
execution_times = np.random.randint(1, 100, num_tasks)
power_consumptions = np.random.randint(1, 50, num_tasks)

tasks = [Task(execution_times[i], power_consumptions[i]) for i in range(num_tasks)]
power_budget = 200

# Schedule tasks and calculate throughput
throughput = schedule_tasks(tasks, power_budget)

# Print the required metric line
print(f"METRIC throughput={throughput}")