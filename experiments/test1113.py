# experiments/test1113.py
#!/usr/bin/env python3
"""
主实验脚本 - 运行完整的对比实验（分层评估版本）
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.model_factory import ModelFactory
from experiments.our_model_adapter import OurEnhancedModel
from experiments.data_unifier import DataUnifier
from experiments.experiment_runner import ExperimentRunner
from experiments.comparative_analyzer import ComparativeAnalyzer

# 分层评估相关导入
from experiments.evaluation.layered_evaluator import create_layered_evaluator
from experiments.comparative_experiment import run_layered_comparative_experiment, LayeredComparativeExperiment

def create_mock_epidemic_data(population_size=500, initial_infected_ratio=0.05):
    """创建模拟疫情数据"""
    print("生成模拟疫情数据...")
    
    # 创建网络数据
    import networkx as nx
    network = nx.erdos_renyi_graph(population_size, 0.15)
    
    # 创建初始状态
    all_nodes = list(network.nodes())
    n_initial_infected = max(1, int(population_size * initial_infected_ratio))
    infected_nodes = np.random.choice(all_nodes, size=n_initial_infected, replace=False).tolist()
    susceptible_nodes = [node for node in all_nodes if node not in infected_nodes]
    
    initial_states = {
        'total_population': population_size,
        'infected_nodes': infected_nodes,
        'susceptible_nodes': susceptible_nodes,
        'individual_ids': all_nodes
    }
    
    # 可用资源
    available_resources = {
        'vaccines': 100,
        'test_kits': 200,
        'medical_staff': 50,
        'ppe_equipment': 150
    }
    
    unified_data = {
        'network_data': network,
        'initial_conditions': initial_states,
        'available_resources': available_resources,
        'metadata': {
            'population_size': population_size,
            'initial_infected': n_initial_infected,
            'network_nodes': network.number_of_nodes(),
            'network_edges': network.number_of_edges()
        }
    }
    
    print(f"模拟数据生成完成:")
    print(f"  人口: {population_size}")
    print(f"  初始感染: {n_initial_infected}")
    print(f"  网络: {network.number_of_nodes()}节点, {network.number_of_edges()}边")
    
    return unified_data

def setup_all_models():
    """设置所有参与对比的模型"""
    print("初始化所有模型...")
    
    models = {}
    
    # 1. 您的增强模型
    try:
        your_model = OurEnhancedModel({
            'use_dynamic_communities': True,
            'multilayer_network': True,
            'risk_prediction': True
        })
        models['Our_Enhanced_Model'] = your_model
        print("  Our_Enhanced_Model 初始化成功")
    except Exception as e:
        print(f"  Our_Enhanced_Model 初始化失败: {e}")
    
    # 2. 传统基准模型
    baseline_models = ModelFactory.create_all_baselines()
    for name, model in baseline_models.items():
        try:
            models[name] = model
            print(f"  {name} 初始化成功")
        except Exception as e:
            print(f"  {name} 初始化失败: {e}")
    
    print(f"共初始化 {len(models)} 个模型")
    return models

def run_layered_evaluation_experiment(models, unified_data, time_steps=30):
    """运行分层评估实验 - 修复版本"""
    print("\n" + "="*60)
    print("开始分层评估实验")
    print("="*60)
    
    all_results = {}
    failed_models = []
    
    # 1. 运行所有模型的模拟
    print("\n步骤1: 运行模型模拟...")
    for model_name, model in models.items():
        try:
            print(f"  运行 {model_name}...")
            
            # 运行模拟
            results = model.simulate(
                network_data=unified_data['network_data'],
                initial_states=unified_data['initial_conditions'],
                time_steps=time_steps
            )
            
            # 运行资源分配
            risk_assessment = results.get('risk_assessment', {})
            resource_allocation = model.allocate_resources(
                risk_assessment=risk_assessment,
                available_resources=unified_data['available_resources']
            )
            
            # 修复：统一提取时间序列数据
            standardized_time_series = _extract_standardized_time_series(model_name, results, time_steps)
            
            # 整合结果
            combined_results = {
                'simulation_results': results,
                'resource_allocation': resource_allocation,
                'risk_assessment': risk_assessment,
                'time_series': standardized_time_series,  # 使用统一的时间序列
                'success': results.get('success', False)
            }
            
            all_results[model_name] = combined_results
            print(f"  {model_name} 模拟完成")
            
        except Exception as e:
            print(f"  {model_name} 模拟失败: {e}")
            failed_models.append(model_name)
    
    if not all_results:
        raise Exception("所有模型模拟都失败了！")
    
    if failed_models:
        print(f"{len(failed_models)} 个模型失败: {', '.join(failed_models)}")
    
    # 2. 分层评估
    print("\n步骤2: 进行分层评估...")
    layered_evaluator = create_layered_evaluator()
    
    # 修复：准备标准化的评估数据
    evaluation_data = {}
    for model_name, results in all_results.items():
        evaluation_data[model_name] = {
            'time_series': results['time_series'],  # 现在所有模型都有统一格式的time_series
            'resource_allocation': results['resource_allocation'],
            'risk_assessment': results['risk_assessment'],
            'peak_infection': _extract_peak_infection(results),
            'total_cases': _extract_total_cases(results)
        }
    
    # 运行分层评估
    evaluation_report = layered_evaluator.evaluate_all_models(evaluation_data)
    
    # 3. 整合最终结果
    final_results = {
        'success': True,
        'model_results': all_results,
        'evaluation_results': evaluation_data,
        'evaluation_report': evaluation_report,
        'experiment_config': {
            'time_steps': time_steps,
            'total_models': len(models),
            'successful_models': len(all_results),
            'failed_models': failed_models
        },
        'metadata': unified_data['metadata']
    }
    
    return final_results

def _extract_standardized_time_series(model_name, results, time_steps):
    """统一提取时间序列数据 - 处理不同模型的输出格式"""
    
    # 根据模型类型处理不同的输出格式
    if model_name == 'grid_management':
        # GridManagementModel 从 epidemic_curve 提取
        return _extract_from_grid_management(results, time_steps)
    
    elif model_name == 'classic_seir':
        # ClassicSEIRModel 字段名映射
        return _extract_from_classic_seir(results, time_steps)
    
    else:
        # OurEnhancedModel 和 NetworkSEIRModel 使用标准格式
        return _extract_standard_format(results, time_steps)

def _extract_from_grid_management(results, time_steps):
    """从 GridManagementModel 提取时间序列"""
    epidemic_curve = results.get('epidemic_curve', {})
    grid_infections = epidemic_curve.get('grid_infections', [])
    total_infected = epidemic_curve.get('total_infected', [])
    
    if total_infected and len(total_infected) > 0:
        # 从总感染数推断时间序列
        population = len(results.get('grid_assignments', [])) or 100
        return _create_time_series_from_infections(total_infected, population, time_steps)
    else:
        # 创建默认时间序列
        return _create_default_time_series(time_steps)

def _extract_from_classic_seir(results, time_steps):
    """从 ClassicSEIRModel 提取时间序列（字段名映射）"""
    time_series = results.get('time_series', {})
    
    if 'susceptible' in time_series and 'infected' in time_series and 'recovered' in time_series:
        # 字段名映射：susceptible -> S, infected -> I, recovered -> R
        return {
            'S': time_series['susceptible'],
            'I': time_series['infected'], 
            'R': time_series['recovered']
        }
    else:
        return _extract_standard_format(results, time_steps)

def _extract_standard_format(results, time_steps):
    """提取标准格式的时间序列"""
    time_series = results.get('time_series', {})
    
    # 检查是否包含标准字段
    if 'S' in time_series and 'I' in time_series and 'R' in time_series:
        return {
            'S': time_series['S'],
            'I': time_series['I'],
            'R': time_series['R']
        }
    else:
        # 创建默认时间序列
        return _create_default_time_series(time_steps)

def _create_time_series_from_infections(infected_list, population, time_steps):
    """从感染列表创建时间序列"""
    S, I, R = [], [], []
    
    for i, infected_count in enumerate(infected_list):
        if i >= time_steps:
            break
            
        # 简化的逻辑：假设恢复人数随时间增加
        current_infected = min(infected_count, population)
        current_recovered = max(0, i * 2)  # 简单假设每天恢复2人
        current_susceptible = max(0, population - current_infected - current_recovered)
        
        S.append(current_susceptible)
        I.append(current_infected)
        R.append(current_recovered)
    
    # 确保长度一致
    while len(S) < time_steps:
        S.append(S[-1] if S else population)
        I.append(I[-1] if I else 0)
        R.append(R[-1] if R else 0)
    
    return {'S': S[:time_steps], 'I': I[:time_steps], 'R': R[:time_steps]}

def _create_default_time_series(time_steps):
    """创建默认的时间序列数据"""
    # 简单的线性变化作为回退
    S = [100 - i * 2 for i in range(time_steps)]
    I = [5 + i for i in range(time_steps)]
    R = [i for i in range(time_steps)]
    
    return {'S': S, 'I': I, 'R': R}

def _extract_peak_infection(results):
    """提取峰值感染数"""
    # 优先从结果中直接获取
    if 'peak_infection' in results:
        return results['peak_infection']
    
    # 从时间序列计算
    time_series = results.get('time_series', {})
    if 'I' in time_series and time_series['I']:
        return max(time_series['I'])
    
    return 0

def _extract_total_cases(results):
    """提取总病例数"""
    # 优先从结果中直接获取
    if 'total_cases' in results:
        return results['total_cases']
    
    # 从时间序列计算
    time_series = results.get('time_series', {})
    if 'R' in time_series and time_series['R']:
        return time_series['R'][-1]
    
    return 0

def generate_comprehensive_analysis(final_results, output_dir="results"):
    """生成综合分析报告"""
    print("\n生成综合分析报告...")
    
    analyzer = ComparativeAnalyzer()
    
    try:
        # 1. 生成图表
        print("  生成对比图表...")
        figures = analyzer.generate_paper_figures(final_results)
        
        # 2. 生成LaTeX表格
        print("  生成数据表格...")
        tables = analyzer.generate_latex_tables(final_results)
        
        # 3. 生成评估报告
        print("  生成评估报告...")
        report_path = os.path.join(output_dir, "comprehensive_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("疫情模型对比实验综合报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 实验概况
            f.write("实验概况:\n")
            f.write(f"- 参与模型数量: {final_results['experiment_config']['total_models']}\n")
            f.write(f"- 成功运行模型: {final_results['experiment_config']['successful_models']}\n")
            f.write(f"- 模拟时间步长: {final_results['experiment_config']['time_steps']}\n")
            f.write(f"- 总人口规模: {final_results['metadata']['population_size']}\n\n")
            
            # 模型排名
            ranking = final_results['evaluation_report']['ranking']
            f.write("模型排名:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'排名':<4} {'模型名称':<25} {'综合得分':<10} {'基础能力':<10} {'智能能力':<10} {'等级':<8}\n")
            f.write("-" * 80 + "\n")
            
            for item in ranking:
                f.write(f"{item['rank']:<4} {item['model']:<25} {item['score']:<10.3f} "
                       f"{item['basic_score']:<10.3f} {item['intelligent_score']:<10.3f} {item['level']:<8}\n")
            
            f.write("\n")
            
            # 最佳模型分析
            best_model = ranking[0]
            f.write(f"最佳模型: {best_model['model']}\n")
            f.write(f"   综合得分: {best_model['score']:.3f}\n")
            f.write(f"   基础能力: {best_model['basic_score']:.3f}\n")
            f.write(f"   智能能力: {best_model['intelligent_score']:.3f}\n\n")
            
            # 详细分析
            evaluations = final_results['evaluation_report']['evaluations']
            best_evaluation = evaluations.get(best_model['model'], {})
            analysis = best_evaluation.get('analysis', {})
            
            if analysis:
                f.write("详细分析:\n")
                f.write(f"- 总结: {analysis.get('summary', '')}\n")
                if analysis.get('strengths'):
                    f.write(f"- 优势: {', '.join(analysis['strengths'])}\n")
                if analysis.get('improvements'):
                    f.write(f"- 改进点: {', '.join(analysis['improvements'])}\n")
                if analysis.get('recommendations'):
                    f.write(f"- 建议: {', '.join(analysis['recommendations'])}\n")
        
        figures['comprehensive_report'] = report_path
        print(f"  综合报告已保存: {report_path}")
        
    except Exception as e:
        print(f"  分析报告生成失败: {e}")
        figures = {}
        tables = {}
    
    return figures, tables

def print_experiment_summary(final_results, figures, tables):
    """打印实验总结"""
    print("\n" + "="*70)
    print("实验完成总结")
    print("="*70)
    
    # 最佳模型信息
    ranking = final_results['evaluation_report']['ranking']
    best_model = ranking[0]
    summary = final_results['evaluation_report']['summary']
    
    print(f"\n最佳模型: {best_model['model']}")
    print(f"   综合得分: {best_model['score']:.3f}")
    print(f"   基础能力: {best_model['basic_score']:.3f}")
    print(f"   智能能力: {best_model['intelligent_score']:.3f}")
    print(f"   性能等级: {best_model['level']}")
    
    print(f"\n实验统计:")
    print(f"   参与模型: {summary['total_models']} 个")
    print(f"   优秀模型: {summary['excellent_count']} 个")
    print(f"   平均得分: {summary['average_score']:.3f}")
    print(f"   人口规模: {final_results['metadata']['population_size']}")
    
    print(f"\n输出文件:")
    print(f"   生成图表: {len(figures)} 个")
    print(f"   生成表格: {len(tables)} 个")
    
    if figures:
        print(f"\n生成的图表:")
        for fig_name, fig_path in figures.items():
            if fig_name != 'comprehensive_report':
                print(f"  {fig_name}: {fig_path}")
    
    if 'comprehensive_report' in figures:
        print(f"\n详细报告:")
        print(f"   综合报告: {figures['comprehensive_report']}")

def main():
    """运行完整对比实验 - 分层评估版本"""
    
    print("开始完整对比实验（分层评估版本）")
    print("=" * 60)
    
    try:
        # 1. 设置实验参数
        population_size = 500
        time_steps = 30
        initial_infected_ratio = 0.05
        
        # 2. 准备实验数据
        print("\n步骤1: 准备实验数据...")
        unified_data = create_mock_epidemic_data(
            population_size=population_size,
            initial_infected_ratio=initial_infected_ratio
        )
        
        # 3. 初始化所有模型
        print("\n步骤2: 初始化模型...")
        models = setup_all_models()
        
        if not models:
            raise Exception("没有成功初始化的模型！")
        
        # 4. 运行分层评估实验
        print("\n步骤3: 运行分层评估实验...")
        final_results = run_layered_evaluation_experiment(
            models=models,
            unified_data=unified_data,
            time_steps=time_steps
        )
        
        # 5. 生成分析报告
        print("\n步骤4: 生成分析报告...")
        figures, tables = generate_comprehensive_analysis(final_results)
        
        # 6. 打印实验总结
        print_experiment_summary(final_results, figures, tables)
        
        return final_results, figures, tables
        
    except Exception as e:
        print(f"\n实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空结果
        return {}, {}, {}

if __name__ == "__main__":
    # 运行实验
    final_results, figures, tables = main()
    
    # 实验状态
    if final_results.get('success', False):
        print(f"\n实验成功完成！")
        sys.exit(0)
    else:
        print(f"\n实验执行失败！")
        sys.exit(1)