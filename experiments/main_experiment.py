"""
主实验脚本 - 运行完整的对比实验
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.model_factory import ModelFactory
from experiments.our_model_adapter import OurEnhancedModel
from experiments.data_unifier import DataUnifier
from experiments.experiment_runner import ExperimentRunner
from experiments.comparative_analyzer import ComparativeAnalyzer

def main():
    """运行完整对比实验"""
    
    print("开始完整对比实验")
    print("=" * 50)
    
    # 1. 初始化组件
    data_unifier = DataUnifier(random_seed=42)
    experiment_runner = ExperimentRunner()
    analyzer = ComparativeAnalyzer()
    
    # 2. 准备统一数据
    print("步骤1: 准备实验数据...")
    mock_data = type('obj', (object,), {'shape': [1000, 10]})()
    unified_data = data_unifier.prepare_unified_data(mock_data, target_size=500)
    
    # 3. 注册所有模型
    print("步骤2: 注册模型...")
    
    # 注册你的模型
    your_model = OurEnhancedModel()
    experiment_runner.register_model("Our_Enhanced_Model", your_model)
    
    # 注册传统模型
    baseline_models = ModelFactory.create_all_baselines()
    for name, model in baseline_models.items():
        experiment_runner.register_model(name, model)
    
    # 4. 运行对比实验
    print("步骤3: 运行对比实验...")
    results = experiment_runner.run_comprehensive_experiment(
        network_data=unified_data['network_data'],
        initial_conditions=unified_data['initial_conditions'],
        available_resources=unified_data['available_resources'],
        time_steps=30,
        random_seed=42
    )
    
    # 5. 生成分析报告和图表
    print("步骤4: 生成分析报告...")
    figures = analyzer.generate_paper_figures(results)
    tables = analyzer.generate_latex_tables(results)
    
    # 6. 打印最终结果
    print("实验完成!")
    
    best_model = results['evaluation']['ranking'][0]['model']
    best_score = results['evaluation']['ranking'][0]['score']
    
    print(f"最佳模型: {best_model} (得分: {best_score:.3f})")
    print(f"生成图表: {len(figures)} 个")
    print(f"生成表格: {len(tables)} 个")

if __name__ == "__main__":
    main()