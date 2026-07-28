import subprocess, os

class DockerSandbox:
    def __init__(self, work_dir="./workspace"):
        self.work_dir = work_dir

    def write_file(self, filename, content):
        with open(os.path.join(self.work_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)

    def read_file(self, filename):
        path = os.path.join(self.work_dir, filename)
        if not os.path.exists(path):
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def run_container_test(self, command):
        absolute_work = os.path.abspath(self.work_dir)
        docker_cmd = ["docker", "run", "--rm", "-v", f"{absolute_work}:/workspace", "-w", "/workspace", "polyglot-research-env", "bash", "-c", command]
        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=90)
            # Use a distinct container-status marker so it can never be confused with the
            # program's own 'SUCCESS' keyword that the peer-review node scans for.
            if result.returncode == 0:
                return f"[CONTAINER_OK]\n{result.stdout}".rstrip()
            return f"[CONTAINER_FAIL]\n{result.stdout}\n{result.stderr}".rstrip()
        except Exception as e:
            return f"CONTAINER EXCEPTION: {str(e)}"
