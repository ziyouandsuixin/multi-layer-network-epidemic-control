#!/usr/bin/env python3
"""
独立参数调优执行脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimization.config_manager import ConfigManager
from optimization.param_optimizer import ParamOptimizer
from optimization.sensitivity_analyzer import SensitivityAnalyzer
from optimization.evaluator import ParameterEvaluator

def main():
    print("开始独立参数调优...")
    
    # 初始化组件
    config_manager = ConfigManager()
    optimizer = ParamOptimizer(config_manager)
    analyzer = SensitivityAnalyzer(optimizer)
    evaluator = ParameterEvaluator()
    
    # 1. 加载当前模拟结果（这里需要从你的模型获取）
    current_simulation_results = {
        'peak_infection_rate': 0.284,
        'recovery_efficiency': 0.15,
        'stability': 0.3,
        'resource_efficiency': 0.957
    }
    
    print("当前模拟结果分析:")
    print(f"  感染峰值: {current_simulation_results['peak_infection_rate']:.3f}")
    print(f"  恢复效率: {current_simulation_results['recovery_efficiency']:.3f}")
    
    # 2. 执行参数优化
    optimization_result = optimizer.auto_tune_based_on_simulation(current_simulation_results)
    
    # 3. 敏感性分析
    param_ranges = {
        'infection.base_prob': [0.01, 0.03, 0.05, 0.08],
        'infection.pressure_weight': [0.01, 0.02, 0.03, 0.05],
        'recovery.base_rate': [0.05, 0.1, 0.15, 0.2]
    }
    sensitivity_results = analyzer.analyze_parameter_sensitivity(param_ranges)
    
    # 4. 保存优化结果
    config_manager.save_optimized_params(
        optimization_result.params, 
        "optimized_v1"
    )
    
    print("参数调优完成!")
    print(f"预期改进: {optimization_result.improvement:.1%}")
    print("优化参数已保存到 configs/optimized_v1_params.yaml")
    
    return optimization_result

if __name__ == "__main__":
    main()