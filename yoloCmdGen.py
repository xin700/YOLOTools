# -*- coding: utf-8 -*-
"""
YOLO命令行生成器
交互式生成Ultralytics YOLO命令，不执行实际操作
支持 YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11
"""

# ==================== YOLO版本配置 ====================
YOLO_VERSIONS = {
    "v5": {
        "name": "YOLOv5",
        "models": {
            "1": ("yolov5n.pt", "Nano (最小最快)"),
            "2": ("yolov5s.pt", "Small"),
            "3": ("yolov5m.pt", "Medium"),
            "4": ("yolov5l.pt", "Large"),
            "5": ("yolov5x.pt", "XLarge (最大最准)"),
        },
        "seg_suffix": "-seg",
        "cls_suffix": "-cls",
        "pose_suffix": None,  # v5不支持pose
    },
    "v8": {
        "name": "YOLOv8",
        "models": {
            "1": ("yolov8n.pt", "Nano (最小最快)"),
            "2": ("yolov8s.pt", "Small"),
            "3": ("yolov8m.pt", "Medium"),
            "4": ("yolov8l.pt", "Large"),
            "5": ("yolov8x.pt", "XLarge (最大最准)"),
        },
        "seg_suffix": "-seg",
        "cls_suffix": "-cls",
        "pose_suffix": "-pose",
    },
    "v9": {
        "name": "YOLOv9",
        "models": {
            "1": ("yolov9t.pt", "Tiny (最小最快)"),
            "2": ("yolov9s.pt", "Small"),
            "3": ("yolov9m.pt", "Medium"),
            "4": ("yolov9c.pt", "Compact"),
            "5": ("yolov9e.pt", "Extended (最大最准)"),
        },
        "seg_suffix": "-seg",
        "cls_suffix": None,
        "pose_suffix": None,
    },
    "v10": {
        "name": "YOLOv10",
        "models": {
            "1": ("yolov10n.pt", "Nano (最小最快)"),
            "2": ("yolov10s.pt", "Small"),
            "3": ("yolov10m.pt", "Medium"),
            "4": ("yolov10b.pt", "Balanced"),
            "5": ("yolov10l.pt", "Large"),
            "6": ("yolov10x.pt", "XLarge (最大最准)"),
        },
        "seg_suffix": None,
        "cls_suffix": None,
        "pose_suffix": None,
    },
    "11": {
        "name": "YOLO11 (最新)",
        "models": {
            "1": ("yolo11n.pt", "Nano (最小最快)"),
            "2": ("yolo11s.pt", "Small"),
            "3": ("yolo11m.pt", "Medium"),
            "4": ("yolo11l.pt", "Large"),
            "5": ("yolo11x.pt", "XLarge (最大最准)"),
        },
        "seg_suffix": "-seg",
        "cls_suffix": "-cls",
        "pose_suffix": "-pose",
    },
}


def clear_screen():
    """清屏"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印标题"""
    print("=" * 60)
    print("      YOLO 命令行生成器 (Ultralytics)")
    print("      支持: YOLOv5 / v8 / v9 / v10 / v11")
    print("=" * 60)
    print()


def get_choice(prompt, options):
    """获取用户选择"""
    while True:
        print(prompt)
        for key, value in options.items():
            print(f"  [{key}] {value}")
        choice = input("\n请输入选项: ").strip()
        if choice in options:
            return choice
        print("无效选项，请重新输入\n")


def get_input(prompt, default=None):
    """获取用户输入，支持默认值"""
    if default:
        user_input = input(f"{prompt} (默认: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def select_yolo_version():
    """选择YOLO版本"""
    version = get_choice("选择YOLO版本:", {
        "1": "YOLOv5  - 经典稳定版本",
        "2": "YOLOv8  - 主流推荐版本",
        "3": "YOLOv9  - 高精度版本",
        "4": "YOLOv10 - 无NMS实时版本",
        "5": "YOLO11  - 最新版本 (2024)",
    })
    version_map = {"1": "v5", "2": "v8", "3": "v9", "4": "v10", "5": "11"}
    return version_map[version]


def select_model(version_key, task_name):
    """根据版本和任务选择模型"""
    version_config = YOLO_VERSIONS[version_key]
    
    # 构建模型选项
    model_options = {}
    for key, (model_file, desc) in version_config["models"].items():
        model_options[key] = f"{model_file} - {desc}"
    model_options[str(len(model_options) + 1)] = "自定义模型路径"
    
    model_choice = get_choice(f"选择{version_config['name']}预训练模型:", model_options)
    
    # 自定义路径
    if model_choice == str(len(version_config["models"]) + 1):
        return get_input("输入模型路径")
    
    # 获取基础模型名
    base_model = version_config["models"][model_choice][0]
    
    # 根据任务类型添加后缀
    if task_name == "segment":
        suffix = version_config.get("seg_suffix")
        if suffix is None:
            print(f"\n⚠️ 警告: {version_config['name']} 不支持分割任务，将使用检测模型")
            return base_model
        return base_model.replace(".pt", f"{suffix}.pt")
    elif task_name == "pose":
        suffix = version_config.get("pose_suffix")
        if suffix is None:
            print(f"\n⚠️ 警告: {version_config['name']} 不支持姿态估计，将使用检测模型")
            return base_model
        return base_model.replace(".pt", f"{suffix}.pt")
    elif task_name == "classify":
        suffix = version_config.get("cls_suffix")
        if suffix is None:
            print(f"\n⚠️ 警告: {version_config['name']} 不支持分类任务，将使用检测模型")
            return base_model
        return base_model.replace(".pt", f"{suffix}.pt")
    else:
        return base_model


def generate_train_command():
    """生成训练命令"""
    print("\n" + "-" * 40)
    print("训练命令配置")
    print("-" * 40 + "\n")
    
    # 选择YOLO版本
    version_key = select_yolo_version()
    print()
    
    # 选择任务类型
    task = get_choice("选择任务类型:", {
        "1": "detect - 目标检测",
        "2": "segment - 实例分割",
        "3": "classify - 图像分类",
        "4": "pose - 姿态估计"
    })
    task_map = {"1": "detect", "2": "segment", "3": "classify", "4": "pose"}
    task_name = task_map[task]
    
    print()
    
    # 选择模型
    model_path = select_model(version_key, task_name)
    
    print()
    
    # 数据集配置
    data_path = get_input("数据集配置文件路径 (data.yaml)")
    
    # 训练参数
    print("\n--- 训练参数 ---\n")
    epochs = get_input("训练轮数 (epochs)", "100")
    imgsz = get_input("图片尺寸 (imgsz)", "640")
    batch = get_input("批次大小 (batch)", "16")
    
    # 高级参数
    print()
    use_advanced = get_choice("是否配置高级参数?", {"y": "是", "n": "否"})
    
    advanced_params = ""
    if use_advanced == "y":
        print("\n--- 高级参数 (直接回车跳过) ---\n")
        
        lr0 = get_input("初始学习率 (lr0)", "")
        if lr0:
            advanced_params += f" lr0={lr0}"
        
        patience = get_input("早停耐心值 (patience)", "")
        if patience:
            advanced_params += f" patience={patience}"
        
        workers = get_input("数据加载线程数 (workers)", "")
        if workers:
            advanced_params += f" workers={workers}"
        
        device_choice = get_choice("运行设备:", {
            "1": "GPU 0",
            "2": "GPU 1", 
            "3": "CPU",
            "4": "多GPU (0,1)",
            "5": "跳过"
        })
        device_map = {"1": "0", "2": "1", "3": "cpu", "4": "0,1"}
        if device_choice != "5":
            advanced_params += f" device={device_map[device_choice]}"
        
        project = get_input("保存目录 (project)", "")
        if project:
            advanced_params += f" project={project}"
        
        name = get_input("实验名称 (name)", "")
        if name:
            advanced_params += f" name={name}"
        
        resume = get_choice("是否断点续训?", {"y": "是", "n": "否"})
        if resume == "y":
            advanced_params += " resume=True"
    
    # 生成命令
    command = f"yolo {task_name} train model={model_path} data={data_path} epochs={epochs} imgsz={imgsz} batch={batch}{advanced_params}"
    
    return command


def generate_predict_command():
    """生成推理命令"""
    print("\n" + "-" * 40)
    print("推理命令配置")
    print("-" * 40 + "\n")
    
    # 模型路径
    model_choice = get_choice("选择模型:", {
        "1": "使用预训练模型",
        "2": "使用自定义模型"
    })
    
    if model_choice == "1":
        # 选择YOLO版本
        version_key = select_yolo_version()
        print()
        
        # 选择任务类型
        task = get_choice("选择任务类型:", {
            "1": "detect - 目标检测",
            "2": "segment - 实例分割",
            "3": "classify - 图像分类",
            "4": "pose - 姿态估计"
        })
        task_map = {"1": "detect", "2": "segment", "3": "classify", "4": "pose"}
        task_name = task_map[task]
        
        print()
        
        # 选择模型
        model_path = select_model(version_key, task_name)
    else:
        # 自定义模型
        task = get_choice("选择任务类型:", {
            "1": "detect - 目标检测",
            "2": "segment - 实例分割",
            "3": "classify - 图像分类",
            "4": "pose - 姿态估计"
        })
        task_map = {"1": "detect", "2": "segment", "3": "classify", "4": "pose"}
        task_name = task_map[task]
        
        print()
        model_path = get_input("输入模型路径 (如 runs/train/exp/weights/best.pt)")
    
    print()
    
    # 输入源
    source_type = get_choice("选择输入源类型:", {
        "1": "图片文件",
        "2": "图片文件夹",
        "3": "视频文件",
        "4": "摄像头",
        "5": "网络流/URL"
    })
    
    if source_type == "4":
        source = "0"
    else:
        source = get_input("输入源路径")
    
    # 推理参数
    print("\n--- 推理参数 ---\n")
    conf = get_input("置信度阈值 (conf)", "0.25")
    iou = get_input("IoU阈值 (iou)", "0.7")
    imgsz = get_input("图片尺寸 (imgsz)", "640")
    
    # 输出选项
    print()
    save = get_choice("是否保存结果?", {"y": "是", "n": "否"})
    save_param = " save=True" if save == "y" else ""
    
    show = get_choice("是否实时显示?", {"y": "是", "n": "否"})
    show_param = " show=True" if show == "y" else ""
    
    # 生成命令
    command = f"yolo {task_name} predict model={model_path} source={source} conf={conf} iou={iou} imgsz={imgsz}{save_param}{show_param}"
    
    return command


def generate_val_command():
    """生成验证命令"""
    print("\n" + "-" * 40)
    print("验证命令配置")
    print("-" * 40 + "\n")
    
    # 选择任务类型
    task = get_choice("选择任务类型:", {
        "1": "detect - 目标检测",
        "2": "segment - 实例分割",
        "3": "classify - 图像分类",
        "4": "pose - 姿态估计"
    })
    task_map = {"1": "detect", "2": "segment", "3": "classify", "4": "pose"}
    task_name = task_map[task]
    
    print()
    
    # 模型路径
    model_path = get_input("模型路径 (如 runs/train/exp/weights/best.pt)")
    
    # 数据集配置
    data_path = get_input("数据集配置文件路径 (data.yaml)")
    
    # 验证参数
    print("\n--- 验证参数 ---\n")
    imgsz = get_input("图片尺寸 (imgsz)", "640")
    batch = get_input("批次大小 (batch)", "16")
    
    # 生成命令
    command = f"yolo {task_name} val model={model_path} data={data_path} imgsz={imgsz} batch={batch}"
    
    return command


def generate_export_command():
    """生成导出命令"""
    print("\n" + "-" * 40)
    print("模型导出配置")
    print("-" * 40 + "\n")
    
    # 模型路径
    model_path = get_input("模型路径 (如 runs/train/exp/weights/best.pt)")
    
    print()
    
    # 导出格式
    format_choice = get_choice("选择导出格式:", {
        "1": "onnx - ONNX格式 (通用)",
        "2": "torchscript - TorchScript",
        "3": "engine - TensorRT (NVIDIA GPU加速)",
        "4": "openvino - OpenVINO (Intel)",
        "5": "coreml - CoreML (Apple)",
        "6": "tflite - TFLite (移动端)"
    })
    format_map = {
        "1": "onnx", "2": "torchscript", "3": "engine",
        "4": "openvino", "5": "coreml", "6": "tflite"
    }
    export_format = format_map[format_choice]
    
    print()
    
    # 导出参数
    imgsz = get_input("图片尺寸 (imgsz)", "640")
    
    # 生成命令
    command = f"yolo export model={model_path} format={export_format} imgsz={imgsz}"
    
    return command


def print_command(command):
    """打印生成的命令"""
    print("\n" + "=" * 60)
    print("生成的命令:")
    print("=" * 60)
    print()
    print(f"  {command}")
    print()
    print("-" * 60)
    print("复制上面的命令到终端运行即可")
    print("-" * 60)


def main():
    """主函数"""
    while True:
        clear_screen()
        print_header()
        
        # 主菜单
        choice = get_choice("请选择操作:", {
            "1": "训练 (train) - 训练模型",
            "2": "推理 (predict) - 使用模型进行预测",
            "3": "验证 (val) - 在验证集上评估模型",
            "4": "导出 (export) - 导出模型格式",
            "q": "退出"
        })
        
        if choice == "q":
            print("\n再见！")
            break
        
        # 根据选择生成命令
        if choice == "1":
            command = generate_train_command()
        elif choice == "2":
            command = generate_predict_command()
        elif choice == "3":
            command = generate_val_command()
        elif choice == "4":
            command = generate_export_command()
        
        # 打印命令
        print_command(command)
        
        # 等待用户
        print()
        input("按 Enter 键继续...")


if __name__ == "__main__":
    main()
