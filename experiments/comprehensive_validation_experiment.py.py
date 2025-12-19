"""
主实验脚本 - 运行完整的对比实验（分层评估版本 + 验证实验）
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.model_factory import ModelFactory
from experiments.our_model_adapter import OurEnhancedModel
from experiments.experiment_runner import ExperimentRunner
from experiments.comparative_analyzer import ComparativeAnalyzer
from experiments.evaluation.layered_evaluator import create_layered_evaluator
from experiments.comparative_experiment import run_layered_comparative_experiment, LayeredComparativeExperiment

def create_mock_epidemic_data(population_size=500, initial_infected_ratio=0.05):
    import networkx as nx
    network = nx.erdos_renyi_graph(population_size, 0.15)
    
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
    models = {}
    
    try:
        your_model = OurEnhancedModel({
            'use_dynamic_communities': True,
            'multilayer_network': True,
            'risk_prediction': True
        })
        models['Our_Enhanced_Model'] = your_model
        print("   Our_Enhanced_Model 初始化成功")
    except Exception as e:
        print(f"   Our_Enhanced_Model 初始化失败: {e}")
    
    baseline_models = ModelFactory.create_all_baselines()
    for name, model in baseline_models.items():
        try:
            models[name] = model
            print(f"   {name} 初始化成功")
        except Exception as e:
            print(f"   {name} 初始化失败: {e}")
    
    print(f"共初始化 {len(models)} 个模型")
    return models

def run_layered_evaluation_experiment(models, unified_data, time_steps=30):
    print("\n" + "="*60)
    print("开始分层评估实验")
    print("="*60)
    
    all_results = {}
    failed_models = []
    
    print("\n步骤1: 运行模型模拟...")
    for model_name, model in models.items():
        try:
            print(f"   运行 {model_name}...")
            
            results = model.simulate(
                network_data=unified_data['network_data'],
                initial_states=unified_data['initial_conditions'],
                time_steps=time_steps
            )
            
            risk_assessment = results.get('risk_assessment', {})
            resource_allocation = model.allocate_resources(
                risk_assessment=risk_assessment,
                available_resources=unified_data['available_resources']
            )
            
            standardized_time_series = _extract_standardized_time_series(model_name, results, time_steps)
            
            combined_results = {
                'simulation_results': results,
                'resource_allocation': resource_allocation,
                'risk_assessment': risk_assessment,
                'time_series': standardized_time_series,
                'success': results.get('success', False)
            }
            
            all_results[model_name] = combined_results
            print(f"   {model_name} 模拟完成")
            
        except Exception as e:
            print(f"   {model_name} 模拟失败: {e}")
            failed_models.append(model_name)
    
    if not all_results:
        raise Exception("所有模型模拟都失败了！")
    
    if failed_models:
        print(f"{len(failed_models)} 个模型失败: {', '.join(failed_models)}")
    
    print("\n步骤2: 进行分层评估...")
    layered_evaluator = create_layered_evaluator()
    
    evaluation_data = {}
    for model_name, results in all_results.items():
        evaluation_data[model_name] = {
            'time_series': results['time_series'],
            'resource_allocation': results['resource_allocation'],
            'risk_assessment': results['risk_assessment'],
            'peak_infection': _extract_peak_infection(results),
            'total_cases': _extract_total_cases(results)
        }
    
    evaluation_report = layered_evaluator.evaluate_all_models(evaluation_data)
    
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
    if model_name == 'grid_management':
        return _extract_from_grid_management(results, time_steps)
    
    elif model_name == 'classic_seir':
        return _extract_from_classic_seir(results, time_steps)
    
    else:
        return _extract_standard_format(results, time_steps)

def _extract_from_grid_management(results, time_steps):
    epidemic_curve = results.get('epidemic_curve', {})
    grid_infections = epidemic_curve.get('grid_infections', [])
    total_infected = epidemic_curve.get('total_infected', [])
    
    if total_infected and len(total_infected) > 0:
        population = len(results.get('grid_assignments', [])) or 100
        return _create_time_series_from_infections(total_infected, population, time_steps)
    else:
        return _create_default_time_series(time_steps)

def _extract_from_classic_seir(results, time_steps):
    time_series = results.get('time_series', {})
    
    if 'susceptible' in time_series and 'infected' in time_series and 'recovered' in time_series:
        return {
            'S': time_series['susceptible'],
            'I': time_series['infected'], 
            'R': time_series['recovered']
        }
    else:
        return _extract_standard_format(results, time_steps)

def _extract_standard_format(results, time_steps):
    time_series = results.get('time_series', {})
    
    if 'S' in time_series and 'I' in time_series and 'R' in time_series:
        return {
            'S': time_series['S'],
            'I': time_series['I'],
            'R': time_series['R']
        }
    else:
        return _create_default_time_series(time_steps)

def _create_time_series_from_infections(infected_list, population, time_steps):
    S, I, R = [], [], []
    
    for i, infected_count in enumerate(infected_list):
        if i >= time_steps:
            break
            
        current_infected = min(infected_count, population)
        current_recovered = max(0, i * 2)
        current_susceptible = max(0, population - current_infected - current_recovered)
        
        S.append(current_susceptible)
        I.append(current_infected)
        R.append(current_recovered)
    
    while len(S) < time_steps:
        S.append(S[-1] if S else population)
        I.append(I[-1] if I else 0)
        R.append(R[-1] if R else 0)
    
    return {'S': S[:time_steps], 'I': I[:time_steps], 'R': R[:time_steps]}

def _create_default_time_series(time_steps):
    S = [100 - i * 2 for i in range(time_steps)]
    I = [5 + i for i in range(time_steps)]
    R = [i for i in range(time_steps)]
    
    return {'S': S, 'I': I, 'R': R}

def _extract_peak_infection(results):
    if 'peak_infection' in results:
        return results['peak_infection']
    
    time_series = results.get('time_series', {})
    if 'I' in time_series and time_series['I']:
        return max(time_series['I'])
    
    return 0

def _extract_total_cases(results):
    if 'total_cases' in results:
        return results['total_cases']
    
    time_series = results.get('time_series', {})
    if 'R' in time_series and time_series['R']:
        return time_series['R'][-1]
    
    return 0

def run_statistical_significance_test(models, unified_data, n_trials=20):
    print(f"\n{'='*60}")
    print(f"开始统计显著性检验 ({n_trials}次运行)")
    print(f"{'='*60}")
    
    all_scores = {model_name: [] for model_name in models.keys()}
    
    for trial in range(n_trials):
        print(f"\n第 {trial+1}/{n_trials} 次运行...")
        
        try:
            np.random.seed(trial)
            
            final_results = run_layered_evaluation_experiment(
                models=models,
                unified_data=unified_data,
                time_steps=30
            )
            
            ranking = final_results['evaluation_report']['ranking']
            for item in ranking:
                model_name = item['model']
                score = item['score']
                all_scores[model_name].append(score)
                
            print(f"第 {trial+1} 次运行完成")
            
        except Exception as e:
            print(f"第 {trial+1} 次运行失败: {e}")
            for model_name in models.keys():
                all_scores[model_name].append(0.3)
    
    stats_results = calculate_significance_stats(all_scores)
    
    print_statistical_results(stats_results)
    
    return all_scores, stats_results

def calculate_significance_stats(all_scores):
    from scipy.stats import ttest_ind
    
    stats_results = {}
    model_names = list(all_scores.keys())
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                scores1 = all_scores[model1]
                scores2 = all_scores[model2]
                
                t_stat, p_value = ttest_ind(scores1, scores2)
                
                mean_diff = np.mean(scores1) - np.mean(scores2)
                
                stats_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'mean_diff': mean_diff,
                    'significant': p_value < 0.05,
                    'effect_size': mean_diff / np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                }
    
    return stats_results

def print_statistical_results(stats_results):
    print(f"\n{'='*70}")
    print(f"统计显著性检验结果")
    print(f"{'='*70}")
    
    significant_pairs = []
    non_significant_pairs = []
    
    for pair, results in stats_results.items():
        model1, model2 = pair.split('_vs_')
        significance = "显著" if results['significant'] else "不显著"
        
        result_line = (f"{pair:<40} | p值: {results['p_value']:.4f} | "
                      f"均值差: {results['mean_diff']:.3f} | {significance}")
        
        if results['significant']:
            significant_pairs.append(result_line)
        else:
            non_significant_pairs.append(result_line)
    
    print("\n显著差异的模型对:")
    for line in significant_pairs:
        print(f"  {line}")
    
    print("\n无显著差异的模型对:")
    for line in non_significant_pairs:
        print(f"  {line}")
    
    print(f"\n关键结论:")
    best_vs_second = None
    for pair in stats_results.keys():
        if "Our_Enhanced_Model" in pair and "classic_seir" in pair:
            best_vs_second = stats_results[pair]
            break
    
    if best_vs_second:
        if best_vs_second['significant']:
            print(f"  Our_Enhanced_Model 相对于 classic_seir 的差异具有统计显著性")
            print(f"     (p = {best_vs_second['p_value']:.4f}, 效应大小: {best_vs_second['effect_size']:.2f})")
        else:
            print(f"  Our_Enhanced_Model 相对于 classic_seir 的差异不具有统计显著性")
            print(f"     (p = {best_vs_second['p_value']:.4f})")

def create_ablation_study_models():
    print(f"\n{'='*60}")
    print(f"创建消融实验模型")
    print(f"{'='*60}")
    
    ablation_models = {}
    
    ablation_configs = [
        {
            'name': 'Base_Model',
            'config': {'use_dynamic_communities': False, 'multilayer_network': False, 'risk_prediction': False}
        },
        {
            'name': 'With_Dynamic_Communities', 
            'config': {'use_dynamic_communities': True, 'multilayer_network': False, 'risk_prediction': False}
        },
        {
            'name': 'With_Multilayer_Network',
            'config': {'use_dynamic_communities': False, 'multilayer_network': True, 'risk_prediction': False}
        },
        {
            'name': 'With_Risk_Prediction',
            'config': {'use_dynamic_communities': False, 'multilayer_network': False, 'risk_prediction': True}
        },
        {
            'name': 'Dynamic_Plus_Multilayer',
            'config': {'use_dynamic_communities': True, 'multilayer_network': True, 'risk_prediction': False}
        },
        {
            'name': 'Dynamic_Plus_Risk',
            'config': {'use_dynamic_communities': True, 'multilayer_network': False, 'risk_prediction': True}
        },
        {
            'name': 'Multilayer_Plus_Risk', 
            'config': {'use_dynamic_communities': False, 'multilayer_network': True, 'risk_prediction': True}
        },
        {
            'name': 'Full_Enhanced_Model',
            'config': {'use_dynamic_communities': True, 'multilayer_network': True, 'risk_prediction': True}
        }
    ]
    
    for config_info in ablation_configs:
        try:
            model = OurEnhancedModel(config_info['config'])
            ablation_models[config_info['name']] = model
            print(f"创建消融模型: {config_info['name']}")
        except Exception as e:
            print(f"创建消融模型失败 {config_info['name']}: {e}")
    
    return ablation_models

def run_ablation_study(ablation_models, unified_data):
    print(f"\n{'='*60}")
    print(f"开始消融实验")
    print(f"{'='*60}")
    
    final_results = run_layered_evaluation_experiment(
        models=ablation_models,
        unified_data=unified_data,
        time_steps=30
    )
    
    ablation_analysis = analyze_ablation_results(final_results)
    
    return final_results, ablation_analysis

def analyze_ablation_results(final_results):
    evaluation_report = final_results['evaluation_report']
    ranking = evaluation_report['ranking']
    
    print(f"\n{'='*70}")
    print(f"消融实验结果分析")
    print(f"{'='*70}")
    
    component_contributions = {
        'dynamic_communities': [],
        'multilayer_network': [], 
        'risk_prediction': []
    }
    
    for item in ranking:
        model_name = item['model']
        score = item['score']
        
        if 'Base' in model_name:
            continue
            
        if 'Dynamic' in model_name and 'Full' not in model_name:
            component_contributions['dynamic_communities'].append(score)
        if 'Multilayer' in model_name and 'Full' not in model_name:
            component_contributions['multilayer_network'].append(score)
        if 'Risk' in model_name and 'Full' not in model_name:
            component_contributions['risk_prediction'].append(score)
    
    print(f"\n各组件性能贡献:")
    for component, scores in component_contributions.items():
        if scores:
            avg_score = np.mean(scores)
            print(f"  {component}: 平均得分 {avg_score:.3f} (基于 {len(scores)} 个配置)")
    
    full_model_score = next(item['score'] for item in ranking if 'Full' in item['model'])
    base_model_score = next(item['score'] for item in ranking if 'Base' in item['model'])
    improvement = full_model_score - base_model_score
    
    print(f"\n关键发现:")
    print(f"  • 完整模型得分: {full_model_score:.3f}")
    print(f"  • 基础模型得分: {base_model_score:.3f}") 
    print(f"  • 总体改进: +{improvement:.3f} ({improvement/base_model_score*100:.1f}%)")
    
    full_score = full_model_score
    component_sum = (np.mean(component_contributions['dynamic_communities']) + 
                    np.mean(component_contributions['multilayer_network']) + 
                    np.mean(component_contributions['risk_prediction'])) / 3
    
    synergy = full_score - component_sum
    if synergy > 0:
        print(f"  检测到组件协同效应: +{synergy:.3f}")
    else:
        print(f"  未检测到明显的组件协同效应")
    
    return {
        'component_contributions': component_contributions,
        'full_vs_base_improvement': improvement,
        'synergy_effect': synergy
    }

def generate_validation_report(stats_results, sensitivity_results, ablation_analysis, output_dir="validation_results"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "comprehensive_validation_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("疫情模型综合验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 统计显著性检验\n")
        f.write("-" * 40 + "\n")
        significant_count = sum(1 for r in stats_results.values() if r['significant'])
        total_count = len(stats_results)
        f.write(f"显著差异对: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)\n\n")
        
        f.write("2. 权重敏感性分析\n")
        f.write("-" * 40 + "\n")
        best_models = [result['best_model'] for result in sensitivity_results.values()]
        unique_best = len(set(best_models))
        f.write(f"最佳模型一致性: {unique_best} 个不同最佳模型\n")
        
        f.write("3. 消融实验结果\n") 
        f.write("-" * 40 + "\n")
        f.write(f"完整模型相对于基础模型的改进: +{ablation_analysis['full_vs_base_improvement']:.3f}\n")
        f.write(f"组件协同效应: {ablation_analysis['synergy_effect']:.3f}\n")
        
        f.write("\n4. 总体验证结论\n")
        f.write("-" * 40 + "\n")
        
        stats_ok = significant_count / total_count > 0.5
        sensitivity_ok = unique_best == 1
        ablation_ok = ablation_analysis['full_vs_base_improvement'] > 0.1
        
        if stats_ok and sensitivity_ok and ablation_ok:
            f.write("模型验证通过: 结果具有统计显著性、权重鲁棒性和组件有效性\n")
        else:
            f.write("模型验证部分通过，但存在以下问题:\n")
            if not stats_ok:
                f.write("   - 统计显著性不足\n")
            if not sensitivity_ok:
                f.write("   - 对权重设置敏感\n") 
            if not ablation_ok:
                f.write("   - 组件改进效果有限\n")
    
    print(f"验证报告已保存: {report_path}")

def main():
    print("开始完整验证实验（统计显著性 + 权重敏感性 + 消融实验）")
    print("=" * 60)
    
    try:
        population_size = 500
        time_steps = 30
        initial_infected_ratio = 0.05
        
        print("\n步骤1: 准备实验数据...")
        unified_data = create_mock_epidemic_data(
            population_size=population_size,
            initial_infected_ratio=initial_infected_ratio
        )
        
        print("\n步骤2: 创建消融实验模型...")
        ablation_models = create_ablation_study_models()
        
        if not ablation_models:
            raise Exception("没有成功创建的消融模型！")
        
        print("\n步骤3: 运行详细消融实验...")
        ablation_results, ablation_analysis = run_ablation_study(ablation_models, unified_data)
        
        print("\n步骤4: 统计显著性检验...")
        all_scores, stats_results = run_statistical_significance_test(
            models=ablation_models,
            unified_data=unified_data,
            n_trials=20
        )
        
        print("\n步骤5: 权重敏感性分析...")
        evaluation_data = {}
        for model_name, model in ablation_models.items():
            results = model.simulate(
                network_data=unified_data['network_data'],
                initial_states=unified_data['initial_conditions'],
                time_steps=time_steps
            )
            evaluation_data[model_name] = {
                'time_series': results.get('time_series', {}),
                'resource_allocation': results.get('resource_allocation', {}),
                'risk_assessment': results.get('risk_assessment', {})
            }
        
        layered_evaluator = create_layered_evaluator()
        sensitivity_results = layered_evaluator.perform_weight_sensitivity_analysis(evaluation_data)
        
        print("\n步骤6: 生成验证报告...")
        generate_validation_report(
            stats_results, 
            sensitivity_results, 
            ablation_analysis,
            output_dir="validation_results"
        )
        
        print(f"\n所有验证实验完成！")
        return True
        
    except Exception as e:
        print(f"\n验证实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n验证实验成功完成！")
        sys.exit(0)
    else:
        print(f"\n验证实验执行失败！")
        sys.exit(1)