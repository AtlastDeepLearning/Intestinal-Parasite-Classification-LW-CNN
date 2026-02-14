import os
import sys
import platform
import subprocess
import shutil
import re
from datetime import datetime

def run_command(command):
    """Runs a shell command and returns the output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running '{command}': {e}"

def get_file_content(filepath):
    """Reads content from a file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return f.read().strip()
    except:
        pass
    return "N/A"

def check_system_info():
    info = []
    info.append(f"Timestamp: {datetime.now()}")
    info.append(f"Hostname: {platform.node()}")
    info.append(f"OS: {platform.system()} {platform.release()}")
    info.append(f"Platform: {platform.platform()}")
    info.append(f"Architecture: {platform.machine()}")
    
    # Raspberry Pi specific model check
    model = get_file_content("/proc/device-tree/model")
    info.append(f"Hardware Model: {model}")
    
    # Kernel version
    info.append(f"Kernel: {run_command('uname -a')}")
    
    return "\n".join(info)

def check_hardware_info():
    info = []
    
    # CPU Info
    info.append("--- CPU Info ---")
    # lscpu might be too verbose, lets try simple proc/cpuinfo parsing or just summary
    cpu_info = run_command("lscpu | grep 'Model name\\|Architecture\\|CPU(s):'")
    if not cpu_info:
        cpu_info = run_command("cat /proc/cpuinfo | grep 'model name' | head -n 1")
    info.append(cpu_info)

    # Memory
    info.append("\n--- Memory Info ---")
    info.append(run_command("free -h"))

    # Disk Usage
    info.append("\n--- Disk Usage ---")
    info.append(run_command("df -h /"))

    # Temperature (Pi specific)
    info.append("\n--- Temperature ---")
    temp = run_command("vcgencmd measure_temp")
    if "command not found" in temp or not temp:
         temp = get_file_content("/sys/class/thermal/thermal_zone0/temp")
         if temp and temp != "N/A":
             try:
                temp = f"Temp: {float(temp)/1000:.1f}'C"
             except:
                pass
    info.append(temp)
    
    # Throttle state
    throttled = run_command("vcgencmd get_throttled")
    if "command not found" not in throttled:
        info.append(f"Throttled State: {throttled}")

    return "\n".join(info)

def check_python_env():
    info = []
    info.append(f"Python Executable: {sys.executable}")
    info.append(f"Python Version: {sys.version}")
    
    # Check if in venv
    in_venv = (sys.prefix != sys.base_prefix)
    info.append(f"Running in Virtual Environment: {in_venv}")
    if in_venv:
        info.append(f"Venv Path: {sys.prefix}")
        
    return "\n".join(info)

def check_dependencies():
    info = []
    info.append("--- Pip Packages ---")
    info.append(run_command(f"{sys.executable} -m pip list"))
    
    info.append("\n--- Apt Packages (Relevant ones) ---")
    # Check for common relevant system libs
    libs = ["libcamera", "python3-opencv", "python3-tensorflow", "python3-tflite-runtime", "hailo-all"]
    for lib in libs:
        res = run_command(f"dpkg -l | grep {lib}")
        if res:
            info.append(res)
    
    return "\n".join(info)

def check_peripherals():
    info = []
    info.append("--- USB Devices ---")
    info.append(run_command("lsusb"))
    
    info.append("\n--- Cameras (libcamera) ---")
    cam_info = run_command("libcamera-hello --list-cameras")
    info.append(cam_info)
    
    return "\n".join(info)

def check_hailo_status():
    info = []
    info.append("--- Hailo AI Accelerator Status ---")
    hailo_scan = run_command("hailortcli scan")
    info.append(f"Hailo Scan:\n{hailo_scan}")
    
    # Check if device exists in /dev
    if os.path.exists("/dev/hailo0"):
        info.append("/dev/hailo0 exists.")
    else:
         info.append("/dev/hailo0 DOES NOT exist.")
         
    return "\n".join(info)

def analyze_compatibility():
    # Simple logic to suggest compatibility based on what we see
    notes = []
    
    py_ver = sys.version_info
    
    # Check Python version compatibility for common libraries
    if py_ver.major == 3 and py_ver.minor >= 11:
        notes.append("NOTE: Python 3.11+ detected. Ensure you are using tflite-runtime or compatible TensorFlow wheels. Official TF builds might be limited.")
        
    pip_list = run_command(f"{sys.executable} -m pip list")
    
    if "tensorflow" in pip_list and "aarch64" not in run_command("uname -m"):
         notes.append("WARNING: TensorFlow found but not on AArch64? (Just a sanity check, usually Pi5 is aarch64)")

    if "tflite-runtime" not in pip_list and "tensorflow" not in pip_list:
        notes.append("MISSING: No TensorFlow or TFLite Runtime found. You need one of these for the parasite classifier.")

    return "\n".join(notes)


def main():
    report_file = "pi_environment_report.txt"
    print(f"Collecting system information... writing to {report_file}")
    
    sections = [
        ("SYSTEM INFORMATION", check_system_info),
        ("HARDWARE INFORMATION", check_hardware_info),
        ("PYTHON ENVIRONMENT", check_python_env),
        ("PYTHON DEPENDENCIES", check_dependencies),
        ("PERIPHERALS", check_peripherals),
        ("HAILO AI ACCELERATOR", check_hailo_status),
        ("COMPATIBILITY ANALYSIS", analyze_compatibility)
    ]
    
    with open(report_file, "w") as f:
        for title, func in sections:
            f.write(f"========================================\n")
            f.write(f"{title}\n")
            f.write(f"========================================\n")
            try:
                content = func()
                f.write(content + "\n\n")
            except Exception as e:
                f.write(f"Error collecting data: {e}\n\n")
            print(f"Finished {title}")
            
    print(f"\nReport generated: {os.path.abspath(report_file)}")
    print("Please review this file to check installed libraries and hardware status.")

if __name__ == "__main__":
    main()
