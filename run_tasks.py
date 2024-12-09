import subprocess

def run_command(command):
    print(f"Running command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' failed with exit status {e.returncode}")
        print(f"Error message: {e.stderr}")
        raise  # Re-raise the exception to stop the script

def main():
    run_command("docker-compose up -d")
    run_command("clearml-init")
    run_command("python gradio_interface.py")

if __name__ == "__main__":
    main()
