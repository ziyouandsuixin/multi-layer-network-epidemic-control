import sys
import os

def check_environment():
    print("=== 环境验证 ===")
    
    python_path = sys.executable
    print(f"Python路径: {python_path}")
    
    if 'venv' in python_path.lower() or 'virtualenv' in python_path.lower():
        print("运行在虚拟环境中")
    else:
        print("可能不在虚拟环境中")
    
    work_dir = os.getcwd()
    print(f"工作目录: {work_dir}")
    
    required_packages = ['pandas', 'numpy', 'networkx', 'scikit-learn']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} 已安装")
        except ImportError:
            print(f"{package} 未安装")
    
    project_files = ['models', 'utils', 'run_simple.py']
    for file in project_files:
        if os.path.exists(file):
            print(f"{file} 存在")
        else:
            print(f"{file} 不存在")
    
    print("\n=== 验证完成 ===")

if __name__ == "__main__":
    check_environment()