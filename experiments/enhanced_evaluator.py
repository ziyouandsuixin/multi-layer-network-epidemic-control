"""
增强版评估器 - 修复数据结构匹配
"""

import json
import yaml
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

class EnhancedEvaluator:
    """增强版评估器 - 修复数据结构匹配"""
    
    def __init__(self, weights_config_path: str = "experiments/config/evaluation_weights.yaml"):
        self.weights = self._load_weights(weights_config_path)
    
    def _load_weights(self, config_path: str) -> Dict[str, Any]:
        """加载评估权重配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_weights()
    
    def _get_default_weights(self) -> Dict[str, Any]:
        """默认评估权重"""
        return {
            'epidemic_control': {'weight': 0.40, 'metrics': {}},
            'resource_efficiency': {'weight': 0.35, 'metrics': {}},
            'socioeconomic_impact': {'weight': 0.25, 'metrics': {}}
        }
    
    def evaluate_comprehensive(self, model_results: Dict[str, Any], 
                             initial_conditions: Dict[str, Any],
                             resource_usage: Dict[str, Any]) -> Dict[str, float]:
        """综合评估模型表现 - 修复数据结构处理"""
        
        scores = {}
        
        # 疫情控制效果评估
        epidemic_score = self._evaluate_epidemic_control(model_results, initial_conditions)
        scores['epidemic_control'] = epidemic_score
        
        # 资源效率评估
        resource_score = self._evaluate_resource_efficiency(model_results, resource_usage)
        scores['resource_efficiency'] = resource_score
        
        # 社会经济影响评估
        socioeconomic_score = self._evaluate_socioeconomic_impact(model_results, initial_conditions)
        scores['socioeconomic_impact'] = socioeconomic_score
        
        # 综合得分
        total_score = (
            epidemic_score * self.weights['epidemic_control']['weight'] +
            resource_score * self.weights['resource_efficiency']['weight'] +
            socioeconomic_score * self.weights['socioeconomic_impact']['weight']
        )
        
        scores['comprehensive_score'] = total_score
        
        return scores
    
    def _evaluate_epidemic_control(self, model_results: Dict, initial_conditions: Dict) -> float:
        """评估疫情控制效果 - 修复数据结构访问"""
        try:
            epidemic_results = model_results.get('epidemic_results', {})
            if not epidemic_results:
                return 0.3
            
            time_series = epidemic_results.get('time_series', {})
            
            if not time_series:
                return 0.3
                
            # 提取关键指标
            if 'infected' in time_series:
                infected_curve = time_series['infected']
            elif 'I' in time_series:
                infected_curve = time_series['I']
            else:
                return 0.3
            
            total_cases = 0
            if 'recovered' in time_series and time_series['recovered']:
                total_cases = time_series['recovered'][-1]
            elif 'R' in time_series and time_series['R']:
                total_cases = time_series['R'][-1]
            elif 'total_cases' in epidemic_results:
                total_cases = epidemic_results['total_cases']
            
            initial_infected = len(initial_conditions.get('infected_nodes', []))
            total_population = initial_conditions.get('total_population', 1)
            
            # 计算指标
            peak_infection = max(infected_curve) if infected_curve else 0
            peak_rate = peak_infection / total_population
            
            final_infected = infected_curve[-1] if infected_curve else 0
            attack_rate = total_cases / total_population
            
            # 理想情况：低峰值、低攻击率
            peak_score = max(0, 1 - peak_rate * 2)
            attack_score = max(0, 1 - attack_rate)
            
            epidemic_score = (peak_score + attack_score) / 2
            return min(1.0, max(0.0, epidemic_score))
            
        except Exception as e:
            return 0.3
    
    def _evaluate_resource_efficiency(self, model_results: Dict, resource_usage: Dict) -> float:
        """评估资源使用效率 - 修复数据结构访问"""
        try:
            constrained_allocation = model_results.get('constrained_allocation', {})
            if not constrained_allocation:
                return 0.5
            
            # 计算资源利用率
            total_allocated = 0
            for community_resources in constrained_allocation.values():
                total_allocated += sum(community_resources.values())
            
            available_resources = resource_usage.get('available_resources', {})
            total_available = sum(available_resources.values())
            
            utilization = total_allocated / total_available if total_available > 0 else 0
            
            # 分配有效性加分
            allocation_valid = model_results.get('allocation_valid', False)
            validity_bonus = 0.2 if allocation_valid else 0
            
            efficiency_score = utilization * 0.6 + 0.2 + validity_bonus
            
            return min(1.0, max(0.0, efficiency_score))
            
        except Exception as e:
            return 0.5
    
    def _evaluate_socioeconomic_impact(self, model_results: Dict, initial_conditions: Dict) -> float:
        """评估社会经济影响 - 修复数据结构访问"""
        try:
            total_population = initial_conditions.get('total_population', 1)
            epidemic_results = model_results.get('epidemic_results', {})
            
            # 估算封控人口比例
            time_series = epidemic_results.get('time_series', {})
            if 'infected' in time_series:
                peak_infection = max(time_series['infected']) if time_series['infected'] else 0
            elif 'I' in time_series:
                peak_infection = max(time_series['I']) if time_series['I'] else 0
            else:
                peak_infection = 0
            
            lockdown_ratio = min(1.0, peak_infection / total_population * 3)
            
            # 估算经济成本
            duration = len(time_series.get('infected', [])) if time_series.get('infected') else 30
            economic_cost = duration * lockdown_ratio * 100
            
            # 理想情况：低成本、低封控比例
            lockdown_score = max(0, 1 - lockdown_ratio)
            cost_score = max(0, 1 - economic_cost / 5000)
            
            socioeconomic_score = (lockdown_score + cost_score) / 2
            return min(1.0, max(0.0, socioeconomic_score))
            
        except Exception as e:
            return 0.5
    
    def generate_comparative_report(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成对比实验报告 - 修复数据结构访问"""
        
        # 直接使用experiment_results
        all_results = experiment_results
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(all_results.keys()),
            'comparative_scores': {},
            'ranking': [],
            'detailed_analysis': {}
        }
        
        # 计算各模型得分
        successful_models = 0
        for model_name, results in all_results.items():
            if results.get('success', False):
                scores = self.evaluate_comprehensive(
                    results,
                    results.get('initial_conditions', {}),
                    results.get('resource_usage', {})
                )
                report['comparative_scores'][model_name] = scores
                successful_models += 1
            else:
                # 为失败的模型提供默认分数
                report['comparative_scores'][model_name] = {
                    'epidemic_control': 0.3,
                    'resource_efficiency': 0.5,
                    'socioeconomic_impact': 0.5,
                    'comprehensive_score': 0.4,
                    'error': results.get('error', '模型运行失败')
                }
        
        # 模型排名（只排成功的模型）
        successful_scores = {
            name: score for name, score in report['comparative_scores'].items()
            if all_results[name].get('success', False)
        }
        
        if successful_scores:
            ranked_models = sorted(
                successful_scores.items(),
                key=lambda x: x[1].get('comprehensive_score', 0),
                reverse=True
            )
            
            report['ranking'] = [{'model': name, 'score': score['comprehensive_score']} 
                               for name, score in ranked_models]
        else:
            report['ranking'] = []
        
        # 详细分析
        report['detailed_analysis'] = self._generate_detailed_analysis(report['comparative_scores'], all_results)
        report['successful_models_count'] = successful_models
        report['total_models_count'] = len(all_results)
        
        return report
    
    def _generate_detailed_analysis(self, scores: Dict[str, Dict], all_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成详细分析"""
        analysis = {
            'strengths_weaknesses': {},
            'recommendations': [],
            'key_insights': [],
            'performance_summary': {}
        }
        
        # 性能总结
        successful_models = [name for name in all_results.keys() if all_results[name].get('success', False)]
        if successful_models:
            best_model = max(successful_models, 
                           key=lambda x: scores[x].get('comprehensive_score', 0))
            best_score = scores[best_model].get('comprehensive_score', 0)
            
            analysis['performance_summary'] = {
                'best_model': best_model,
                'best_score': best_score,
                'successful_count': len(successful_models),
                'total_count': len(all_results)
            }
        
        for model_name, model_scores in scores.items():
            strengths = []
            weaknesses = []
            
            epidemic_score = model_scores.get('epidemic_control', 0)
            resource_score = model_scores.get('resource_efficiency', 0)
            socio_score = model_scores.get('socioeconomic_impact', 0)
            
            # 优势分析
            if epidemic_score > 0.7:
                strengths.append("优秀的疫情控制能力")
            if resource_score > 0.7:
                strengths.append("高效的资源利用")
            if socio_score > 0.7:
                strengths.append("良好的社会经济影响")
            
            # 劣势分析
            if epidemic_score < 0.4:
                weaknesses.append("疫情控制效果不佳")
            if resource_score < 0.4:
                weaknesses.append("资源利用效率低")
            if socio_score < 0.4:
                weaknesses.append("社会经济成本高")
            
            analysis['strengths_weaknesses'][model_name] = {
                'strengths': strengths,
                'weaknesses': weaknesses,
                'success': all_results[model_name].get('success', False)
            }
        
        # 生成建议和洞察
        if successful_models:
            analysis['recommendations'].append("建议采用综合表现最佳的模型进行实际应用")
            analysis['key_insights'].append("网络结构对传播动态有显著影响")
        
        if len(successful_models) > 1:
            analysis['key_insights'].append("不同模型在疫情控制和资源效率间存在权衡")
        
        return analysis
    
    def save_evaluation_report(self, report: Dict[str, Any], output_path: str = None):
        """保存评估报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/comparative_experiments/evaluation_report_{timestamp}.json"
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path