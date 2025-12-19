#D:\whu\coursers\Social\1109\evaluation\performance_evaluator.py
'''

防控效果评估系统

功能概述：
评估公共卫生干预措施的综合效果，包括：
- 疫情控制效果（峰值降低、持续时间缩短等）
- 资源使用效率（成本效益、目标精准度等）  
- 社会经济影响（经济成本、社会干扰等）
- 与基线方案的对比分析

输入：资源分配方案 + 基线方案
输出：综合评估得分和详细指标
"""

class InterventionEvaluator:
    """干预效果评估器
    
    主要功能：
    1. evaluate_intervention() - 主评估函数
    2. 疫情控制效果评估
    3. 资源效率评估  
    4. 社会经济影响评估
    5. 与基线方案对比
    6. 计算综合得分
    
    在D:\whu\coursers\Social\1109\run_complete_system.py
    的第五部分：5. 效果评估
    使用示例：
    >>> evaluator = InterventionEvaluator()
    >>> results = evaluator.evaluate_intervention(allocation_plan, baseline_plan, simulation_data)
    >>> print(f"综合得分: {results['overall_score']:.3f}")

'''
"""
防控效果评估系统
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class InterventionEvaluator:
    """干预效果评估器"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_intervention(self, allocation_plan, baseline_plan, simulation_data):
        """评估干预效果"""
        print("评估干预效果...")
        
        evaluation_results = {
            'epidemic_control': self._evaluate_epidemic_control(allocation_plan, simulation_data),
            'resource_efficiency': self._evaluate_resource_efficiency(allocation_plan, baseline_plan),
            'socioeconomic_impact': self._evaluate_socioeconomic_impact(allocation_plan),
            'comparative_analysis': self._compare_with_baseline(allocation_plan, baseline_plan)
        }
        
        # 计算综合得分
        evaluation_results['overall_score'] = self._calculate_overall_score(evaluation_results)
        
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'results': evaluation_results
        })
        
        return evaluation_results
    
    def _evaluate_epidemic_control(self, allocation_plan, simulation_data):
        """评估疫情控制效果"""
        control_metrics = {
            'peak_reduction': 0.0,  # 峰值降低
            'duration_reduction': 0.0,  # 持续时间缩短
            'total_cases_reduction': 0.0,  # 总病例减少
            'r_effective': 0.0  # 有效再生数
        }
        
        # 基于资源分配估计防控效果
        total_efficiency = allocation_plan['estimated_impact']
        
        # 模拟防控效果（简化模型）
        control_metrics['peak_reduction'] = total_efficiency * 0.6
        control_metrics['duration_reduction'] = total_efficiency * 0.4
        control_metrics['total_cases_reduction'] = total_efficiency * 0.7
        control_metrics['r_effective'] = max(0.5, 1.5 - total_efficiency)  # 降低再生数
        
        return control_metrics
    
    def _evaluate_resource_efficiency(self, allocation_plan, baseline_plan):
        """评估资源使用效率"""
        efficiency_metrics = {
            'resource_utilization': 0.0,
            'cost_effectiveness': 0.0,
            'targeting_accuracy': 0.0
        }
        
        # 资源利用率
        efficiency_metrics['resource_utilization'] = allocation_plan['resource_efficiency']
        
        # 成本效益（简化计算）
        total_impact = allocation_plan['estimated_impact']
        estimated_cost = self._estimate_intervention_cost(allocation_plan)
        efficiency_metrics['cost_effectiveness'] = total_impact / max(estimated_cost, 1)
        
        # 目标精准度（高风险社区资源占比）
        high_risk_resources = sum(
            sum(comm['resource_allocation'].values()) 
            for comm in allocation_plan['high_risk_communities']
        )
        total_resources = sum(
            sum(comm['resource_allocation'].values()) 
            for comm in allocation_plan['high_risk_communities'] + 
                       allocation_plan['medium_risk_communities'] + 
                       allocation_plan['low_risk_communities']
        )
        
        efficiency_metrics['targeting_accuracy'] = high_risk_resources / total_resources if total_resources > 0 else 0
        
        return efficiency_metrics
    
    def _evaluate_socioeconomic_impact(self, allocation_plan):
        """评估社会经济影响"""
        socioeconomic_metrics = {
            'economic_cost': 0.0,
            'social_disruption': 0.0,
            'public_acceptance': 0.0
        }
        
        # 经济成本估计
        total_cost = self._estimate_intervention_cost(allocation_plan)
        socioeconomic_metrics['economic_cost'] = total_cost
        
        # 社会干扰（与干预强度相关）
        intervention_intensity = allocation_plan['estimated_impact']
        socioeconomic_metrics['social_disruption'] = intervention_intensity * 0.3
        
        # 公众接受度（与目标精准度负相关）
        socioeconomic_metrics['public_acceptance'] = 0.8 - (socioeconomic_metrics['social_disruption'] * 0.5)
        
        return socioeconomic_metrics
    
    def _estimate_intervention_cost(self, allocation_plan):
        """估计干预成本 - 修复版本"""
        cost_rates = {
            'vaccines': 200,  # 每剂成本
            'test_kits': 6,  # 每个检测试剂盒成本
            'medical_staff': 200,  # 每位医护人员日成本
            'masks':   0.1 #每个口罩成本
        }
        
        total_cost = 0
        
        # 处理不同的分配方案结构
        for risk_level in ['high', 'medium', 'low']:
            communities_key = f'{risk_level}_risk_communities'
            
            if communities_key in allocation_plan:
                # 我们的分配方案结构
                communities = allocation_plan[communities_key]
                for community in communities:
                    if 'resource_allocation' in community:
                        for resource, amount in community['resource_allocation'].items():
                            if resource in cost_rates:
                                total_cost += amount * cost_rates[resource]
            else:
                # 基线方案或其他结构
                continue
        
        # 如果没有计算到成本，使用默认估算
        if total_cost == 0:
            total_cost = 5000  # 默认成本
        
        return total_cost
    
    def _compare_with_baseline(self, allocation_plan, baseline_plan):
        """与基线方案比较 - 修复版本"""
        comparison = {
            'improvement_in_efficiency': 0.0,
            'improvement_in_impact': 0.0,
            'cost_savings': 0.0
        }
        
        # 效率提升
        our_efficiency = allocation_plan.get('resource_efficiency', 0)
        baseline_efficiency = baseline_plan.get('resource_efficiency', 0.5)
        comparison['improvement_in_efficiency'] = our_efficiency - baseline_efficiency
        
        # 影响提升
        our_impact = allocation_plan.get('estimated_impact', 0)
        baseline_impact = baseline_plan.get('estimated_impact', 0.4)
        comparison['improvement_in_impact'] = our_impact - baseline_impact
        
        # 成本节约 - 使用简化的成本估算
        our_cost = self._estimate_intervention_cost(allocation_plan)
        
        # 基线成本估算（简化版）
        if isinstance(baseline_plan, dict) and any(key.endswith('_risk_communities') for key in baseline_plan.keys()):
            # 如果基线方案有完整的社区结构
            baseline_cost = self._estimate_intervention_cost(baseline_plan)
        else:
            # 简化估算：假设基线方案成本比我们高20%
            baseline_cost = our_cost * 1.2
        
        comparison['cost_savings'] = baseline_cost - our_cost
        
        return comparison
    
    def _calculate_overall_score(self, evaluation_results):
        """修复综合得分计算 - 优化版本"""
        # 确保所有指标都有合理值
        control_score = max(0, np.mean(list(evaluation_results['epidemic_control'].values())))
        efficiency_score = max(0, evaluation_results['resource_efficiency'].get('resource_utilization', 0))
        
        return (control_score * 0.6 + efficiency_score * 0.4)