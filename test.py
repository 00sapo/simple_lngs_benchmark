import time
import os
import multiprocessing
import argparse
import math
import shutil
import statistics
from scipy import stats

import torch

HAS_GPU = torch.cuda.is_available()


def cpu_test(iterations, complexity):
    """
    Test CPU performance by performing mathematical operations.

    Args:
        iterations: Number of iterations to run
        complexity: Complexity level (1-5)
    """
    start_time = time.time()

    # Adjust complexity factor
    complexity_factor = complexity * 500000

    for _ in range(iterations):
        # Perform CPU-intensive calculations
        result = 0
        for i in range(complexity_factor):
            result += math.sin(i) * math.cos(i)
            if i % 10000 == 0:
                result = math.sqrt(abs(result))

    elapsed = time.time() - start_time
    return elapsed


def disk_io_test(file_size_mb, block_size_kb, operation_type):
    """
    Test disk I/O performance by writing/reading files.

    The write operation uses a file 100x smaller than the read operation
    to reduce wear on storage devices and speed up the benchmark.

    Args:
        file_size_mb: Size of test file in MB (for read operations)
        block_size_kb: Size of each block in KB
        operation_type: "write", "read", or "both"
    """
    test_dir = "benchmark_test_dir"
    test_file = os.path.join(test_dir, "test_file.dat")

    # Create test directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    block_size = block_size_kb * 1024
    read_file_size = file_size_mb * 1024 * 1024
    write_file_size = read_file_size // 100  # Write file is 100x smaller

    read_blocks = read_file_size // block_size
    write_blocks = write_file_size // block_size

    results = {}

    # Write test
    if operation_type in ["write", "both"]:
        start_time = time.time()
        with open(test_file, "wb") as f:
            for _ in range(write_blocks):
                f.write(os.urandom(block_size))
                f.flush()
                os.fsync(f.fileno())

        write_time = time.time() - start_time
        # Calculate speed based on the actual size written
        write_speed = (write_file_size / 1024 / 1024) / write_time
        results["write"] = {"time": write_time, "speed": write_speed}

    # Read test
    if operation_type in ["read", "both"]:
        # For read test, we need a larger file
        # If file exists from write test, delete it first as it's too small
        if os.path.exists(test_file):
            os.remove(test_file)

        # Create a larger file for read testing
        with open(test_file, "wb") as f:
            for _ in range(read_blocks):
                f.write(os.urandom(block_size))

        # Clear cache to ensure accurate read testing (Linux only)
        if os.name == "posix":
            os.system("sync")
            try:
                # Try to clear page cache on Linux
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3")
            except:
                pass

        start_time = time.time()
        with open(test_file, "rb") as f:
            for _ in range(read_blocks):
                _ = f.read(block_size)  # Read but don't store in an unused variable

        read_time = time.time() - start_time
        # Calculate speed based on the actual size read
        read_speed = file_size_mb / read_time
        results["read"] = {"time": read_time, "speed": read_speed}

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

    return results


def gpu_test(iterations, complexity):
    """
    Test GPU performance by performing tensor operations.

    Args:
        iterations: Number of iterations to run
        complexity: Complexity level (1-5)

    Returns:
        elapsed_time: Time taken to complete the test
        has_gpu: Whether a GPU was detected and used
    """
    if not HAS_GPU:
        return None, False

    start_time = time.time()

    # Adjust complexity factor
    matrix_size = 1000 * complexity

    # Move computation to GPU
    device = torch.device("cuda")

    d = torch.rand(matrix_size, matrix_size, device=device)
    for _ in range(iterations):
        # Create random matrices
        a = torch.rand(matrix_size, matrix_size, device=device)
        b = torch.rand(matrix_size, matrix_size, device=device)

        # Matrix multiplication (compute-intensive operation)
        c = torch.matmul(a, b) + d

        # Add some more operations
        d = torch.sin(c) + torch.cos(c)

        # Ensure operation completes before timing
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    return elapsed, True


def calculate_statistics(results):
    """
    Calculate mean, standard deviation, and 95% confidence interval
    for a list of benchmark results.

    Args:
        results: List of tuples (run_index, value)

    Returns:
        dict: Dictionary containing statistical measures
    """
    if not results:
        return {}

    # Extract just the values for statistical calculations
    values = [v for _, v in results]

    mean = statistics.mean(values)

    # Find min and max with their indices
    min_value = min(values)
    max_value = max(values)
    min_index = next(idx for idx, val in results if val == min_value)
    max_index = next(idx for idx, val in results if val == max_value)

    if len(values) > 1:
        stdev = statistics.stdev(values)
        sem = stdev / math.sqrt(len(values))
        # Calculate 95% confidence interval
        ci_95 = stats.t.ppf(0.975, len(values) - 1) * sem
        return {
            "mean": mean,
            "stdev": stdev,
            "ci_95_low": mean - ci_95,
            "ci_95_high": mean + ci_95,
            "min": {"value": min_value, "run": min_index},
            "max": {"value": max_value, "run": max_index},
            "samples": len(values),
        }
    else:
        return {
            "mean": mean,
            "stdev": 0,
            "ci_95_low": mean,
            "ci_95_high": mean,
            "min": {"value": min_value, "run": min_index},
            "max": {"value": max_value, "run": max_index},
            "samples": 1,
        }


def run_single_benchmark(
    cpu_strength, disk_strength, gpu_strength=0, operation_type="both"
):
    """Run a single benchmark with the specified parameters"""
    results = {}

    # Run CPU benchmark
    if cpu_strength > 0:
        cpu_time = cpu_test(cpu_strength, cpu_strength)
        results["cpu"] = {"time": cpu_time}

    # Run disk benchmark
    if disk_strength > 0:
        settings = {
            "file_size_mb": 200 * disk_strength,
            "block_size_kb": 2 ** round(disk_strength + 3),  # go from 2^3 to 2^12
        }

        disk_results = disk_io_test(
            settings["file_size_mb"], settings["block_size_kb"], operation_type
        )
        results["disk"] = disk_results

    # Run GPU benchmark
    if gpu_strength > 0:
        gpu_time, gpu_available = gpu_test(gpu_strength * 3, gpu_strength)

        if gpu_available:
            results["gpu"] = {"time": gpu_time}
        else:
            results["gpu"] = {"error": "No compatible GPU found"}

    return results


def run_benchmark(
    cpu_strength, disk_strength, gpu_strength=0, operation_type="both", num_runs=10
):
    """Run benchmarks multiple times and calculate statistics"""

    # Initialize result storage - store (run_index, value) tuples to track which run produced min/max
    all_results = {
        "cpu": {"times": []},
        "disk": {
            "write": {"times": [], "speeds": []},
            "read": {"times": [], "speeds": []},
        },
        "gpu": {"times": []},
    }

    # Display test parameters
    print(f"\nRunning {num_runs} benchmark iterations...")

    if cpu_strength > 0:
        print(f"CPU benchmark (strength: {cpu_strength})")
        print(
            f"- Performing {cpu_strength} iterations at complexity level {cpu_strength}"
        )

    if disk_strength > 0:
        file_size_mb = 200 * disk_strength
        block_size_kb = 2 ** round(disk_strength + 3)
        print(f"Disk I/O benchmark (strength: {disk_strength})")
        print(f"- Testing with {file_size_mb} MB file, {block_size_kb} KB blocks")

    if gpu_strength > 0:
        if HAS_GPU:
            print(f"GPU benchmark (strength: {gpu_strength})")
            print(
                f"- Performing {gpu_strength * 3} iterations at complexity level {gpu_strength}"
            )
        else:
            print("No compatible GPU found. Skipping GPU test.")

    # Run benchmarks multiple times
    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}...", end="", flush=True)

        # Run the benchmark
        result = run_single_benchmark(
            cpu_strength, disk_strength, gpu_strength, operation_type
        )

        # Store CPU results with run index
        if "cpu" in result:
            all_results["cpu"]["times"].append((i + 1, result["cpu"]["time"]))

        # Store disk results with run index
        if "disk" in result:
            if "write" in result["disk"]:
                all_results["disk"]["write"]["times"].append(
                    (i + 1, result["disk"]["write"]["time"])
                )
                all_results["disk"]["write"]["speeds"].append(
                    (i + 1, result["disk"]["write"]["speed"])
                )

            if "read" in result["disk"]:
                all_results["disk"]["read"]["times"].append(
                    (i + 1, result["disk"]["read"]["time"])
                )
                all_results["disk"]["read"]["speeds"].append(
                    (i + 1, result["disk"]["read"]["speed"])
                )

        # Store GPU results with run index
        if "gpu" in result and "time" in result["gpu"]:
            all_results["gpu"]["times"].append((i + 1, result["gpu"]["time"]))

        print(" done")

    # Calculate statistics
    stats_results = {}
    if all_results["cpu"]["times"]:
        stats_results["cpu"] = {
            "time": calculate_statistics(all_results["cpu"]["times"])
        }

    if all_results["disk"]["write"]["times"]:
        stats_results["disk_write"] = {
            "time": calculate_statistics(all_results["disk"]["write"]["times"]),
            "speed": calculate_statistics(all_results["disk"]["write"]["speeds"]),
        }

    if all_results["disk"]["read"]["times"]:
        stats_results["disk_read"] = {
            "time": calculate_statistics(all_results["disk"]["read"]["times"]),
            "speed": calculate_statistics(all_results["disk"]["read"]["speeds"]),
        }

    if all_results["gpu"]["times"]:
        stats_results["gpu"] = {
            "time": calculate_statistics(all_results["gpu"]["times"])
        }

    # Print results with confidence intervals, min, and max
    print("\n=== Benchmark Results ===")

    if "cpu" in stats_results:
        cpu_stats = stats_results["cpu"]["time"]
        print(f"\nCPU Performance:")
        print(f"- Average: {cpu_stats['mean']:.2f} seconds")
        print(
            f"- 95% Confidence Interval: {cpu_stats['ci_95_low']:.2f} to {cpu_stats['ci_95_high']:.2f} seconds"
        )
        print(
            f"- Min: {cpu_stats['min']['value']:.2f} seconds (Run {cpu_stats['min']['run']})"
        )
        print(
            f"- Max: {cpu_stats['max']['value']:.2f} seconds (Run {cpu_stats['max']['run']})"
        )
        print(f"- Standard Deviation: {cpu_stats['stdev']:.2f} seconds")

    if "disk_write" in stats_results:
        write_time_stats = stats_results["disk_write"]["time"]
        write_speed_stats = stats_results["disk_write"]["speed"]
        print(f"\nDisk Write Performance:")
        print(f"- Average Time: {write_time_stats['mean']:.2f} seconds")
        print(
            f"- 95% CI Time: {write_time_stats['ci_95_low']:.2f} to {write_time_stats['ci_95_high']:.2f} seconds"
        )
        print(
            f"- Min Time: {write_time_stats['min']['value']:.2f} seconds (Run {write_time_stats['min']['run']})"
        )
        print(
            f"- Max Time: {write_time_stats['max']['value']:.2f} seconds (Run {write_time_stats['max']['run']})"
        )
        print(f"- Average Speed: {write_speed_stats['mean']:.2f} MB/s")
        print(
            f"- 95% CI Speed: {write_speed_stats['ci_95_low']:.2f} to {write_speed_stats['ci_95_high']:.2f} MB/s"
        )
        print(
            f"- Min Speed: {write_speed_stats['min']['value']:.2f} MB/s (Run {write_speed_stats['min']['run']})"
        )
        print(
            f"- Max Speed: {write_speed_stats['max']['value']:.2f} MB/s (Run {write_speed_stats['max']['run']})"
        )

    if "disk_read" in stats_results:
        read_time_stats = stats_results["disk_read"]["time"]
        read_speed_stats = stats_results["disk_read"]["speed"]
        print(f"\nDisk Read Performance:")
        print(f"- Average Time: {read_time_stats['mean']:.2f} seconds")
        print(
            f"- 95% CI Time: {read_time_stats['ci_95_low']:.2f} to {read_time_stats['ci_95_high']:.2f} seconds"
        )
        print(
            f"- Min Time: {read_time_stats['min']['value']:.2f} seconds (Run {read_time_stats['min']['run']})"
        )
        print(
            f"- Max Time: {read_time_stats['max']['value']:.2f} seconds (Run {read_time_stats['max']['run']})"
        )
        print(f"- Average Speed: {read_speed_stats['mean']:.2f} MB/s")
        print(
            f"- 95% CI Speed: {read_speed_stats['ci_95_low']:.2f} to {read_speed_stats['ci_95_high']:.2f} MB/s"
        )
        print(
            f"- Min Speed: {read_speed_stats['min']['value']:.2f} MB/s (Run {read_speed_stats['min']['run']})"
        )
        print(
            f"- Max Speed: {read_speed_stats['max']['value']:.2f} MB/s (Run {read_speed_stats['max']['run']})"
        )

    if "gpu" in stats_results:
        gpu_stats = stats_results["gpu"]["time"]
        print(f"\nGPU Performance:")
        print(f"- Average: {gpu_stats['mean']:.2f} seconds")
        print(
            f"- 95% Confidence Interval: {gpu_stats['ci_95_low']:.2f} to {gpu_stats['ci_95_high']:.2f} seconds"
        )
        print(
            f"- Min: {gpu_stats['min']['value']:.2f} seconds (Run {gpu_stats['min']['run']})"
        )
        print(
            f"- Max: {gpu_stats['max']['value']:.2f} seconds (Run {gpu_stats['max']['run']})"
        )
        print(f"- Standard Deviation: {gpu_stats['stdev']:.2f} seconds")

    print("\nBenchmark complete!")
    return stats_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU, Disk I/O, and GPU Performance Benchmark"
    )

    parser.add_argument(
        "--cpu",
        type=int,
        choices=range(0, 11),
        default=3,
        help="CPU benchmark strength (0-10, where 0 disables CPU test and 10 is highest)",
    )

    parser.add_argument(
        "--disk",
        type=int,
        choices=range(0, 11),
        default=3,
        help="Disk benchmark strength (0-10, where 0 disables disk test and 10 is highest)",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        choices=range(0, 11),
        default=3,
        help="GPU benchmark strength (0-10, where 0 disables GPU test and 10 is highest)",
    )

    parser.add_argument(
        "--disk-op",
        choices=["read", "write", "both"],
        default="both",
        help="Disk operation to test (read, write, or both)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of times to run each benchmark for statistical analysis",
    )

    parser.add_argument(
        "--clean", action="store_true", help="Clean up any leftover benchmark files"
    )

    args = parser.parse_args()

    # Clean up if requested
    if args.clean:
        if os.path.exists("benchmark_test_dir"):
            shutil.rmtree("benchmark_test_dir")
        print("Cleaned up benchmark files.")
        exit(0)

    # Display system info
    print("\n=== System Information ===")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")

    os.system("uname -a")

    # Display GPU info if PyTorch is available
    if HAS_GPU:
        print("\n=== GPU Information ===")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA Version: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
        )

    # Run benchmark
    run_benchmark(args.cpu, args.disk, args.gpu, args.disk_op, args.runs)
