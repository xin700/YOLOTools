# -*- coding: utf-8 -*-
"""
机器学习实战课程 - 环境检测脚本
用于检测所有必需的库是否已正确安装
"""

import sys

def check_environment():
    """检测所有必需的库是否已安装"""
    
    print("=" * 50)
    print("     机器学习实战课程 - 环境检测")
    print("=" * 50)
    print()
    
    # 需要检测的库列表
    required_packages = [
        ("torch", "PyTorch - 深度学习框架"),
        ("torchvision", "TorchVision - 计算机视觉工具"),
        ("matplotlib", "Matplotlib - 绘图库"),
        ("numpy", "NumPy - 数值计算库"),
        ("cv2", "OpenCV - 图像处理库"),
        ("PIL", "Pillow - 图像处理库"),
        ("ultralytics", "Ultralytics - YOLOv8框架"),
        ("IPython", "IPython - Jupyter内核"),
        ("ipykernel", "ipykernel - Jupyter内核插件"),
    ]
    
    errors = []
    success = []
    
    # 逐个检测库
    for package_name, description in required_packages:
        try:
            module = __import__(package_name)
            
            # 获取版本号
            version = getattr(module, "__version__", "未知版本")
            
            success.append((package_name, description, version))
            print(f"✓ {description}")
            print(f"  包名: {package_name}, 版本: {version}")
            
        except ImportError as e:
            errors.append((package_name, description, str(e)))
            print(f"✗ {description}")
            print(f"  包名: {package_name}")
            print(f"  错误: {e}")
        
        print()
    
    # 检测系统是否有NVIDIA GPU
    def check_nvidia_gpu():
        """检测系统是否有NVIDIA GPU"""
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    has_nvidia_gpu = check_nvidia_gpu()
    
    # 检测PyTorch GPU支持
    print("-" * 50)
    print("GPU 检测:")
    print("-" * 50)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        # 显示PyTorch版本信息
        print(f"  PyTorch版本: {torch.__version__}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA 可用")
            print(f"  GPU数量: {gpu_count}")
            print(f"  GPU名称: {gpu_name}")
            print(f"  CUDA版本: {torch.version.cuda}")
            
            # 检测cuDNN版本
            if torch.backends.cudnn.is_available():
                print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
                print(f"  cuDNN启用: {torch.backends.cudnn.enabled}")
        else:
            if has_nvidia_gpu:
                print("⚠ 检测到NVIDIA GPU，但当前安装的是CPU版本PyTorch！")
                print()
                print("  请安装CUDA版本的PyTorch以启用GPU加速:")
                print("  --------------------------------------------------")
                print("  pip uninstall torch torchvision torchaudio -y")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                print("  --------------------------------------------------")
                print()
                print("  注: CUDA版本必须从PyTorch官方源下载，国内镜像仅提供CPU版本")
            else:
                print("⚠ 未检测到NVIDIA GPU，将使用CPU运行")
                print("  (CPU也可以完成所有实验，只是速度较慢)")
    except:
        if has_nvidia_gpu:
            print("⚠ 检测到NVIDIA GPU，但PyTorch未安装")
            print()
            print("  请安装CUDA版本的PyTorch:")
            print("  --------------------------------------------------")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("  --------------------------------------------------")
            print()
            print("  注: CUDA版本必须从PyTorch官方源下载，国内镜像仅提供CPU版本")
        else:
            print("⚠ 无法检测GPU状态")
    
    print()
    
    # 输出总结
    print("=" * 50)
    print("检测结果总结")
    print("=" * 50)
    print()
    
    if errors:
        print(f"❌ 检测失败！有 {len(errors)} 个库未安装：")
        print()
        for package_name, description, error in errors:
            print(f"  - {package_name} ({description})")
        
        print()
        print("请使用以下命令安装缺失的库：")
        print()
        print("  pip install torch torchvision matplotlib numpy opencv-python pillow ultralytics")
        print()
        print("或者分别安装：")
        for package_name, _, _ in errors:
            # 处理特殊包名
            pip_name = package_name
            if package_name == "cv2":
                pip_name = "opencv-python"
            elif package_name == "PIL":
                pip_name = "pillow"
            print(f"  pip install {pip_name}")
        
        return False
    else:
        print(f"✅ 恭喜！所有 {len(success)} 个必需库均已正确安装！")
        print()
        print("已安装的库：")
        for package_name, description, version in success:
            print(f"  - {package_name} ({version})")
        print()
        print("环境准备就绪，可以开始课程学习！")
        return True


if __name__ == "__main__":
    result = check_environment()
    print()
    print("=" * 50)
    
    # 等待用户按键退出（方便直接双击运行时查看结果）
    input("按 Enter 键退出...")
