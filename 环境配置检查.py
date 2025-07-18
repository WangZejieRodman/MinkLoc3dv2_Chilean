#!/usr/bin/env python3
"""
MinkLoc3Dv2环境完整检查脚本
检查CUDA、PyTorch、MinkowskiEngine及相关依赖的安装和兼容性

使用方法:
    python check_minkloc3dv2_env.py
    或
    python /home/wzj/pan1/MinkLoc3Dv2/check_minkloc3dv2_env.py
"""

import sys
import os
import subprocess
import platform
import traceback
from datetime import datetime

class Colors:
    """控制台颜色代码"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title):
    """打印格式化的标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(message):
    """打印成功信息"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """打印错误信息"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    """打印信息"""
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.END}")

def run_command(command):
    """安全地运行系统命令"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timeout", 1
    except Exception as e:
        return "", str(e), 1

def check_basic_environment():
    """1. 基础环境检查"""
    print_header("1. 基础环境检查")
    
    results = []
    
    # 操作系统信息
    try:
        os_info = platform.platform()
        print_success(f"操作系统: {os_info}")
        results.append(("OS", True, os_info))
    except Exception as e:
        print_error(f"获取操作系统信息失败: {e}")
        results.append(("OS", False, str(e)))
    
    # Python版本
    try:
        python_version = sys.version
        print_success(f"Python版本: {python_version}")
        results.append(("Python", True, python_version))
    except Exception as e:
        print_error(f"获取Python版本失败: {e}")
        results.append(("Python", False, str(e)))
    
    # Conda环境
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'N/A')
        if conda_env != 'N/A':
            print_success(f"Conda环境: {conda_env}")
            results.append(("Conda Env", True, conda_env))
        else:
            print_warning("未检测到Conda环境")
            results.append(("Conda Env", False, "Not detected"))
    except Exception as e:
        print_error(f"检查Conda环境失败: {e}")
        results.append(("Conda Env", False, str(e)))
    
    # 环境变量检查
    env_vars = ['CUDA_HOME', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS']
    for var in env_vars:
        try:
            value = os.environ.get(var, 'Not set')
            if value != 'Not set':
                print_success(f"{var}: {value}")
                results.append((var, True, value))
            else:
                print_warning(f"{var}: 未设置")
                results.append((var, False, "Not set"))
        except Exception as e:
            print_error(f"检查{var}失败: {e}")
            results.append((var, False, str(e)))
    
    return results

def check_cuda_pytorch():
    """2. CUDA和PyTorch兼容性检查"""
    print_header("2. CUDA和PyTorch兼容性检查")
    
    results = []
    
    # 检查nvcc
    try:
        stdout, stderr, returncode = run_command("nvcc --version")
        if returncode == 0 and "release" in stdout:
            cuda_version = stdout.split("release")[1].split(",")[0].strip()
            print_success(f"NVCC版本: CUDA {cuda_version}")
            results.append(("NVCC", True, cuda_version))
        else:
            print_error("NVCC未找到或版本获取失败")
            results.append(("NVCC", False, stderr))
    except Exception as e:
        print_error(f"检查NVCC失败: {e}")
        results.append(("NVCC", False, str(e)))
    
    # 检查nvidia-smi
    try:
        stdout, stderr, returncode = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")
        if returncode == 0:
            gpu_info = stdout.split('\n')[0] if stdout else "Unknown GPU"
            print_success(f"GPU信息: {gpu_info}")
            results.append(("GPU Info", True, gpu_info))
        else:
            print_error("无法获取GPU信息")
            results.append(("GPU Info", False, stderr))
    except Exception as e:
        print_error(f"检查GPU信息失败: {e}")
        results.append(("GPU Info", False, str(e)))
    
    # 检查PyTorch
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_version_torch = torch.version.cuda
        print_success(f"PyTorch版本: {pytorch_version}")
        print_success(f"PyTorch编译的CUDA版本: {cuda_version_torch}")
        results.append(("PyTorch", True, pytorch_version))
        results.append(("PyTorch CUDA", True, cuda_version_torch))
        
        # CUDA可用性
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            compute_capability = torch.cuda.get_device_capability(current_device)
            
            print_success(f"CUDA可用: True")
            print_success(f"GPU数量: {device_count}")
            print_success(f"当前GPU: {device_name}")
            print_success(f"计算能力: {compute_capability[0]}.{compute_capability[1]}")
            
            results.append(("CUDA Available", True, "True"))
            results.append(("GPU Count", True, str(device_count)))
            results.append(("GPU Name", True, device_name))
            results.append(("Compute Capability", True, f"{compute_capability[0]}.{compute_capability[1]}"))
            
            # 基本CUDA操作测试
            try:
                x = torch.tensor([1.0, 2.0, 3.0]).cuda()
                y = x * 2
                result = y.cpu().numpy()
                print_success(f"CUDA操作测试: 通过 {result}")
                results.append(("CUDA Operations", True, "Passed"))
            except Exception as e:
                print_error(f"CUDA操作测试失败: {e}")
                results.append(("CUDA Operations", False, str(e)))
        else:
            print_error("CUDA不可用")
            results.append(("CUDA Available", False, "False"))
    
    except ImportError as e:
        print_error(f"PyTorch未安装: {e}")
        results.append(("PyTorch", False, str(e)))
    except Exception as e:
        print_error(f"PyTorch检查失败: {e}")
        results.append(("PyTorch", False, str(e)))
    
    return results

def check_minkowski_engine():
    """3. MinkowskiEngine安装检查和基本功能测试"""
    print_header("3. MinkowskiEngine安装检查和功能测试")
    
    results = []
    
    # 导入测试
    try:
        import MinkowskiEngine as ME
        me_version = ME.__version__
        print_success(f"MinkowskiEngine版本: {me_version}")
        results.append(("MinkowskiEngine Import", True, me_version))
        
        # CUDA支持检查
        try:
            cuda_support = ME.is_cuda_available()
            if cuda_support:
                print_success("MinkowskiEngine CUDA支持: 可用")
                results.append(("ME CUDA Support", True, "Available"))
            else:
                print_warning("MinkowskiEngine CUDA支持: 不可用")
                results.append(("ME CUDA Support", False, "Not available"))
        except Exception as e:
            print_error(f"检查MinkowskiEngine CUDA支持失败: {e}")
            results.append(("ME CUDA Support", False, str(e)))
        
        # 后端模块检查
        try:
            import MinkowskiEngine.MinkowskiEngineBackend._C as _C
            print_success("MinkowskiEngine后端加载: 成功")
            results.append(("ME Backend", True, "Loaded"))
        except ImportError as e:
            print_warning(f"MinkowskiEngine后端加载: 部分问题 ({e})")
            results.append(("ME Backend", False, str(e)))
        except Exception as e:
            print_error(f"MinkowskiEngine后端检查失败: {e}")
            results.append(("ME Backend", False, str(e)))
        
        # 基本功能测试
        print_info("开始MinkowskiEngine基本功能测试...")
        
        # 测试1: 创建SparseTensor
        try:
            import torch
            coords = torch.IntTensor([
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 1, 1, 1]
            ])
            feats = torch.FloatTensor([
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ])
            
            sparse_tensor = ME.SparseTensor(feats, coords)
            print_success(f"SparseTensor创建: 成功 (shape: {sparse_tensor.shape})")
            results.append(("SparseTensor Creation", True, f"Shape: {sparse_tensor.shape}"))
        except Exception as e:
            print_error(f"SparseTensor创建失败: {e}")
            results.append(("SparseTensor Creation", False, str(e)))
            return results
        
        # 测试2: GPU传输
        if torch.cuda.is_available():
            try:
                feats_gpu = feats.cuda()
                coords_gpu = coords.cuda()
                sparse_tensor_gpu = ME.SparseTensor(feats_gpu, coords_gpu)
                print_success(f"GPU SparseTensor: 成功 (device: {sparse_tensor_gpu.device})")
                results.append(("GPU SparseTensor", True, str(sparse_tensor_gpu.device)))
            except Exception as e:
                print_error(f"GPU SparseTensor创建失败: {e}")
                results.append(("GPU SparseTensor", False, str(e)))
        
        # 测试3: 卷积层
        try:
            conv = ME.MinkowskiConvolution(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                dimension=3
            )
            print_success("MinkowskiConvolution创建: 成功")
            results.append(("MinkowskiConvolution", True, "Created"))
            
            # CPU卷积测试
            try:
                output = conv(sparse_tensor)
                print_success(f"CPU卷积计算: 成功 (output shape: {output.shape})")
                results.append(("CPU Convolution", True, f"Output shape: {output.shape}"))
            except Exception as e:
                print_error(f"CPU卷积计算失败: {e}")
                results.append(("CPU Convolution", False, str(e)))
            
            # GPU卷积测试
            if torch.cuda.is_available():
                try:
                    conv_gpu = conv.cuda()
                    output_gpu = conv_gpu(sparse_tensor_gpu)
                    print_success(f"GPU卷积计算: 成功 (output shape: {output_gpu.shape})")
                    results.append(("GPU Convolution", True, f"Output shape: {output_gpu.shape}"))
                except Exception as e:
                    print_error(f"GPU卷积计算失败: {e}")
                    results.append(("GPU Convolution", False, str(e)))
            
        except Exception as e:
            print_error(f"MinkowskiConvolution创建失败: {e}")
            results.append(("MinkowskiConvolution", False, str(e)))
        
        # 测试4: 其他核心层
        try:
            # 批归一化
            bn = ME.MinkowskiBatchNorm(2)
            print_success("MinkowskiBatchNorm创建: 成功")
            
            # 池化层
            pool = ME.MinkowskiGlobalMaxPooling()
            print_success("MinkowskiGlobalMaxPooling创建: 成功")
            
            # ReLU
            relu = ME.MinkowskiReLU()
            print_success("MinkowskiReLU创建: 成功")
            
            results.append(("Other ME Layers", True, "BatchNorm, Pooling, ReLU"))
            
        except Exception as e:
            print_error(f"其他MinkowskiEngine层创建失败: {e}")
            results.append(("Other ME Layers", False, str(e)))
    
    except ImportError as e:
        print_error(f"MinkowskiEngine未安装: {e}")
        results.append(("MinkowskiEngine Import", False, str(e)))
    except Exception as e:
        print_error(f"MinkowskiEngine检查失败: {e}")
        results.append(("MinkowskiEngine Import", False, str(e)))
    
    return results

def check_dependencies():
    """4. 依赖包检查"""
    print_header("4. 依赖包检查")
    
    results = []
    
    # 定义需要检查的包
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('MinkowskiEngine', 'MinkowskiEngine'),
        ('pytorch_metric_learning', 'pytorch_metric_learning'),
        ('wandb', 'wandb'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pexpect', 'pexpect'),
        ('IPython', 'IPython'),
        ('jupyter', 'jupyter'),
    ]
    
    print_info("检查必需的Python包...")
    
    for display_name, import_name in packages:
        try:
            module = __import__(import_name)
            if hasattr(module, '__version__'):
                version = module.__version__
                print_success(f"{display_name}: {version}")
                results.append((display_name, True, version))
            else:
                print_success(f"{display_name}: 已安装 (无版本信息)")
                results.append((display_name, True, "Installed"))
        except ImportError as e:
            print_error(f"{display_name}: 未安装 ({e})")
            results.append((display_name, False, str(e)))
        except Exception as e:
            print_error(f"{display_name}: 检查失败 ({e})")
            results.append((display_name, False, str(e)))
    
    # 检查系统依赖
    print_info("检查系统依赖...")
    
    system_deps = [
        ("ninja", "ninja --version"),
        ("g++", "g++ --version"),
        ("make", "make --version")
    ]
    
    for dep_name, command in system_deps:
        try:
            stdout, stderr, returncode = run_command(command)
            if returncode == 0:
                version_line = stdout.split('\n')[0] if stdout else "Unknown version"
                print_success(f"{dep_name}: {version_line}")
                results.append((f"{dep_name} (system)", True, version_line))
            else:
                print_warning(f"{dep_name}: 未找到或版本获取失败")
                results.append((f"{dep_name} (system)", False, "Not found"))
        except Exception as e:
            print_warning(f"{dep_name}: 检查失败 ({e})")
            results.append((f"{dep_name} (system)", False, str(e)))
    
    return results

def generate_summary_report(all_results):
    """生成总结报告"""
    print_header("🎯 环境检查总结报告")
    
    total_checks = 0
    passed_checks = 0
    critical_issues = []
    warnings = []
    
    # 统计结果
    for category, results in all_results.items():
        for check_name, status, details in results:
            total_checks += 1
            if status:
                passed_checks += 1
            else:
                # 判断是否为关键问题
                if check_name in ['PyTorch', 'CUDA Available', 'MinkowskiEngine Import', 'SparseTensor Creation']:
                    critical_issues.append(f"{check_name}: {details}")
                else:
                    warnings.append(f"{check_name}: {details}")
    
    # 打印统计信息
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"\n📊 检查统计:")
    print(f"   总检查项: {total_checks}")
    print(f"   通过检查: {passed_checks}")
    print(f"   成功率: {success_rate:.1f}%")
    
    # 环境状态判断
    if len(critical_issues) == 0:
        print_success("🎉 环境配置状态: 优秀！可以正常使用MinkLoc3Dv2")
        status = "EXCELLENT"
    elif len(critical_issues) <= 2:
        print_warning("⚠️ 环境配置状态: 良好，但有一些需要注意的问题")
        status = "GOOD"
    else:
        print_error("❌ 环境配置状态: 存在关键问题，需要修复")
        status = "NEEDS_FIX"
    
    # 打印问题详情
    if critical_issues:
        print(f"\n🚨 关键问题 ({len(critical_issues)}):")
        for issue in critical_issues:
            print_error(f"   • {issue}")
    
    if warnings:
        print(f"\n⚠️ 警告和建议 ({len(warnings)}):")
        for warning in warnings[:5]:  # 只显示前5个警告
            print_warning(f"   • {warning}")
        if len(warnings) > 5:
            print_info(f"   ... 还有 {len(warnings) - 5} 个其他警告")
    
    # 推荐操作
    print(f"\n💡 推荐操作:")
    if status == "EXCELLENT":
        print_success("   ✨ 环境配置完美！可以开始使用MinkLoc3Dv2进行开发")
        print_info("   📚 建议: 查看MinkLoc3Dv2文档开始您的项目")
    elif status == "GOOD":
        print_info("   🔧 建议修复警告中的问题以获得最佳性能")
        print_info("   📝 大部分功能应该可以正常使用")
    else:
        print_error("   🛠️ 必须修复关键问题才能正常使用")
        print_info("   📞 考虑重新运行安装脚本或寻求技术支持")
    
    return status, critical_issues, warnings

def save_detailed_report(all_results, status, critical_issues, warnings):
    """保存详细报告到文件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"环境配置检查报告.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MinkLoc3Dv2 环境检查详细报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"环境状态: {status}\n\n")
            
            # 详细结果
            for category, results in all_results.items():
                f.write(f"\n{category}\n")
                f.write("-" * len(category) + "\n")
                for check_name, status_bool, details in results:
                    status_str = "✅ PASS" if status_bool else "❌ FAIL"
                    f.write(f"{status_str} {check_name}: {details}\n")
            
            # 问题总结
            if critical_issues:
                f.write(f"\n关键问题:\n")
                for issue in critical_issues:
                    f.write(f"  • {issue}\n")
            
            if warnings:
                f.write(f"\n警告:\n")
                for warning in warnings:
                    f.write(f"  • {warning}\n")
        
        print_info(f"详细报告已保存到: {report_file}")
        
    except Exception as e:
        print_warning(f"保存报告失败: {e}")

def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.PURPLE}")
    print("🔍 MinkLoc3Dv2 环境完整检查脚本")
    print("=" * 60)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print(f"脚本位置: {os.path.abspath(__file__)}")
    print("=" * 60)
    print(f"{Colors.END}")
    
    # 收集所有检查结果
    all_results = {}
    
    try:
        # 1. 基础环境检查
        all_results["基础环境检查"] = check_basic_environment()
        
        # 2. CUDA和PyTorch检查
        all_results["CUDA和PyTorch兼容性"] = check_cuda_pytorch()
        
        # 3. MinkowskiEngine检查
        all_results["MinkowskiEngine功能测试"] = check_minkowski_engine()
        
        # 4. 依赖包检查
        all_results["依赖包检查"] = check_dependencies()
        
        # 5. 生成总结报告
        status, critical_issues, warnings = generate_summary_report(all_results)
        
        # 6. 保存详细报告
        save_detailed_report(all_results, status, critical_issues, warnings)
        
        print(f"\n{Colors.BOLD}{Colors.PURPLE}检查完成! 🎊{Colors.END}")
        
        # 返回退出代码
        if status == "EXCELLENT":
            return 0
        elif status == "GOOD":
            return 1
        else:
            return 2
            
    except KeyboardInterrupt:
        print_error("\n用户中断检查")
        return 130
    except Exception as e:
        print_error(f"\n检查过程中发生错误: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
