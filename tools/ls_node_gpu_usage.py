import subprocess


def get_gpu_usage(hostname, timeout=5):
    """
    Connects to the given hostname via SSH, runs `nvidia-smi` to fetch GPU utilization,
    and returns the total VRAM usage.
    """
    try:
        print(f"Processing {hostname}...")  # Report the node being processed
        # Run the `nvidia-smi` command over SSH with a timeout
        command = f"ssh -o ConnectTimeout={timeout} {hostname} nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                timeout=timeout)

        if result.returncode != 0:
            print(f"Error connecting to {hostname}: {result.stderr.strip()}")
            return None

        # Parse GPU memory usage
        output = result.stdout.strip()
        if output:
            gpu_usages = list(map(int, output.splitlines()))
            total_vram_usage = sum(gpu_usages)
            return total_vram_usage
        else:
            return 0
    except subprocess.TimeoutExpired:
        print(f"Timeout while connecting to {hostname}. Skipping.")
        return None
    except Exception as e:
        print(f"Error processing {hostname}: {e}")
        return None


def main():
    # Cluster details
    base_hostname = "login"  # Adjust as per your cluster's hostname pattern
    num_nodes = 40  # Number of login nodes

    # Fetch VRAM usage for each node
    vram_usage_list = []
    for i in range(1, num_nodes + 1):
        # Format hostname: prepend '0' for single-digit node numbers
        node_number = f"{i:02}"  # Format as two digits, e.g., 01, 02, ..., 40
        hostname = f"{base_hostname}{node_number}"
        vram_usage = get_gpu_usage(hostname)
        if vram_usage is not None:
            vram_usage_list.append((hostname, vram_usage))

    # Sort by VRAM usage
    sorted_vram_usage = sorted(vram_usage_list, key=lambda x: x[1], reverse=False)

    # Print results
    print("GPU VRAM Usage (sorted):")
    for hostname, usage in sorted_vram_usage:
        print(f"{hostname}: {usage} MB")


if __name__ == "__main__":
    main()
