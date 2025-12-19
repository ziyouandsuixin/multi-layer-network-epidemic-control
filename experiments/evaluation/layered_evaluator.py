# experiments/evaluation/layered_evaluator.py
import os
import yaml
import numpy as np
from typing import Dict, Any, List
from .basic_evaluator import BasicCapabilityEvaluator, create_basic_evaluator
from .intelligent_evaluator import IntelligentCapabilityEvaluator, create_intelligent_evaluator

class LayeredEvaluator:
    """分层评估器 - 整合基础能力和智能能力"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.basic_evaluator = create_basic_evaluator(config_path)
        self.intelligent_evaluator = create_intelligent_evaluator(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载评估配置"""
        default_config = {
            'layer_weights': {
                'basic': 0.40,
                'intelligent': 0.60
            },
            'ranking_thresholds': {
                'excellent': 0.8,
                'good': 0.6,
                'fair': 0.4,
                'poor': 0.0
            }
        }
        return default_config
    
    def evaluate_single_model(self, model_name: str, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个模型"""
        print(f"\n{'='*50}")
        print(f"评估模型: {model_name}")
        print(f"{'='*50}")
        
        evaluation = {'model_name': model_name}
        
        try:
            # 基础能力评估
            print("📊 进行基础能力评估...")
            basic_scores = self.basic_evaluator.evaluate(model_results)
            evaluation['basic_capability'] = basic_scores
            
            # 智能能力评估
            print("进行智能能力评估...")
            intelligent_scores = self.intelligent_evaluator.evaluate(model_name, model_results)
            evaluation['intelligent_capability'] = intelligent_scores
            
            # 计算综合得分
            final_score = self._calculate_final_score(basic_scores, intelligent_scores)
            evaluation['final_score'] = final_score
            
            # 性能等级
            evaluation['performance_level'] = self._get_performance_level(final_score)
            
            # 生成分析报告
            evaluation['analysis'] = self._generate_analysis(evaluation)
            
            print(f"✅ {model_name} 评估完成: {final_score:.3f} ({evaluation['performance_level']})")
            
        except Exception as e:
            print(f"{model_name} 评估失败: {e}")
            evaluation = self._create_fallback_evaluation(model_name)
            
        return evaluation
    
    def evaluate_all_models(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """评估所有模型并排名"""
        print(f"\n{'='*60}")
        print(f"开始批量评估 {len(all_results)} 个模型")
        print(f"{'='*60}")
        
        evaluations = {}
        
        for model_name, results in all_results.items():
            evaluation = self.evaluate_single_model(model_name, results)
            evaluations[model_name] = evaluation
        
        # 生成排名
        ranked_models = self._rank_models(evaluations)
        
        final_report = {
            'evaluations': evaluations,
            'ranking': ranked_models,
            'summary': self._generate_summary(ranked_models)
        }
        
        self._print_comprehensive_report(final_report)
        
        return final_report
    
    def perform_weight_sensitivity_analysis(self, evaluation_data, weight_combinations=None):
        """
        权重敏感性分析 - 测试不同权重配置的排名稳定性
        """
        if weight_combinations is None:
            weight_combinations = [
                (0.3, 0.7),  # 倾向智能能力
                (0.4, 0.6),  # 当前设置
                (0.5, 0.5),  # 平衡设置
                (0.6, 0.4)   # 倾向基础能力
            ]
        
        print(f"\n{'='*60}")
        print(f"开始权重敏感性分析")
        print(f"{'='*60}")
        
        sensitivity_results = {}
        original_weights = self.config['layer_weights'].copy()
        
        for i, (basic_w, intelligent_w) in enumerate(weight_combinations):
            print(f"\n测试权重组合 {i+1}/{len(weight_combinations)}: {basic_w}:{intelligent_w}")
            
            # 临时修改权重
            self.config['layer_weights'] = {'basic': basic_w, 'intelligent': intelligent_w}
            
            # 使用现有评估逻辑
            evaluation_report = self.evaluate_all_models(evaluation_data)
            
            # 记录排名
            ranking = [item['model'] for item in evaluation_report['ranking']]
            sensitivity_results[f"{basic_w}:{intelligent_w}"] = {
                'ranking': ranking,
                'best_model': ranking[0],
                'scores': {item['model']: item['score'] for item in evaluation_report['ranking']}
            }
        
        # 恢复原始权重
        self.config['layer_weights'] = original_weights
        
        # 分析敏感性
        self._analyze_sensitivity_results(sensitivity_results)
        
        return sensitivity_results
    
    def _analyze_sensitivity_results(self, sensitivity_results):
        """分析权重敏感性结果"""
        print(f"\n{'='*70}")
        print(f"权重敏感性分析结果")
        print(f"{'='*70}")
        
        # 检查最佳模型的稳定性
        best_models = [result['best_model'] for result in sensitivity_results.values()]
        unique_best_models = set(best_models)
        
        print(f"\n最佳模型稳定性分析:")
        for weight, result in sensitivity_results.items():
            print(f"  权重 {weight}: 最佳模型 = {result['best_model']}")
        
        if len(unique_best_models) == 1:
            print(f"\n优秀: 最佳模型在所有权重配置下保持一致")
            print(f"   稳定最佳模型: {list(unique_best_models)[0]}")
        else:
            print(f"\n注意: 最佳模型随权重变化")
            print(f"   出现的所有最佳模型: {', '.join(unique_best_models)}")
        
        # 排名一致性分析
        print(f"\n排名一致性分析:")
        all_models = set()
        for result in sensitivity_results.values():
            all_models.update(result['ranking'])
        
        ranking_consistency = {}
        for model in all_models:
            ranks = []
            for weight, result in sensitivity_results.items():
                if model in result['ranking']:
                    ranks.append(result['ranking'].index(model) + 1)
            ranking_consistency[model] = {
                'mean_rank': np.mean(ranks),
                'std_rank': np.std(ranks),
                'min_rank': min(ranks),
                'max_rank': max(ranks)
            }
        
        for model, consistency in sorted(ranking_consistency.items(), key=lambda x: x[1]['mean_rank']):
            stability = "高" if consistency['std_rank'] < 1.0 else "中" if consistency['std_rank'] < 2.0 else "低"
            print(f"  {model:<25}: 平均排名 {consistency['mean_rank']:.1f} ± {consistency['std_rank']:.1f} "
                  f"(范围: {consistency['min_rank']}-{consistency['max_rank']}) - 稳定性: {stability}")
    
    def _calculate_final_score(self, basic_scores: Dict, intelligent_scores: Dict) -> float:
        """计算最终得分"""
        basic_weight = self.config['layer_weights']['basic']
        intelligent_weight = self.config['layer_weights']['intelligent']
        
        basic_total = basic_scores.get('total_basic_score', 0)
        intelligent_total = intelligent_scores.get('total_intelligent_score', 0)
        
        return (basic_total * basic_weight + intelligent_total * intelligent_weight)
    
    def _get_performance_level(self, score: float) -> str:
        """获取性能等级"""
        thresholds = self.config['ranking_thresholds']
        
        if score >= thresholds['excellent']:
            return '优秀'
        elif score >= thresholds['good']:
            return '良好'
        elif score >= thresholds['fair']:
            return '一般'
        else:
            return '待改进'
    
    def _generate_analysis(self, evaluation: Dict) -> Dict[str, Any]:
        """生成详细分析"""
        model_name = evaluation['model_name']
        basic_scores = evaluation['basic_capability']
        intelligent_scores = evaluation['intelligent_capability']
        final_score = evaluation['final_score']
        
        analysis = {
            'summary': f"{model_name} 综合得分: {final_score:.3f}",
            'strengths': [],
            'improvements': [],
            'recommendations': []
        }
        
        # 分析优势
        if basic_scores.get('epidemic_control', 0) > 0.7:
            analysis['strengths'].append("疫情控制效果显著")
        if basic_scores.get('resource_efficiency', 0) > 0.7:
            analysis['strengths'].append("资源利用高效")
        if intelligent_scores.get('risk_prediction', 0) > 0.6:
            analysis['strengths'].append("具备风险预测能力")
        if intelligent_scores.get('targeted_intervention', 0) > 0.6:
            analysis['strengths'].append("支持精准干预")
        
        # 分析改进点
        if basic_scores.get('epidemic_control', 0) < 0.5:
            analysis['improvements'].append("疫情控制能力需提升")
        if basic_scores.get('resource_efficiency', 0) < 0.5:
            analysis['improvements'].append("资源效率有待优化")
        if intelligent_scores.get('total_intelligent_score', 0) < 0.4:
            analysis['improvements'].append("智能能力需要加强")
        
        # 生成建议
        if not analysis['strengths']:
            analysis['recommendations'].append("需要全面提升模型能力")
        elif analysis['improvements']:
            analysis['recommendations'].append(f"建议重点改进: {', '.join(analysis['improvements'])}")
        else:
            analysis['recommendations'].append("模型表现均衡，继续保持优秀表现")
        
        return analysis
    
    def _rank_models(self, evaluations: Dict[str, Dict]) -> List[Dict]:
        """对模型进行排名"""
        ranked = []
        
        for model_name, evaluation in evaluations.items():
            ranked.append({
                'model': model_name,
                'score': evaluation['final_score'],
                'level': evaluation['performance_level'],
                'basic_score': evaluation['basic_capability']['total_basic_score'],
                'intelligent_score': evaluation['intelligent_capability']['total_intelligent_score']
            })
        
        # 按分数降序排列
        ranked.sort(key=lambda x: x['score'], reverse=True)
        
        # 添加排名
        for i, item in enumerate(ranked):
            item['rank'] = i + 1
        
        return ranked
    
    def _generate_summary(self, ranked_models: List[Dict]) -> Dict[str, Any]:
        """生成总结报告"""
        if not ranked_models:
            return {}
            
        best_model = ranked_models[0]
        avg_score = sum(item['score'] for item in ranked_models) / len(ranked_models)
        
        return {
            'best_model': best_model['model'],
            'best_score': best_model['score'],
            'average_score': avg_score,
            'total_models': len(ranked_models),
            'excellent_count': sum(1 for item in ranked_models if item['level'] == '优秀'),
            'good_count': sum(1 for item in ranked_models if item['level'] == '良好')
        }
    
    def _print_comprehensive_report(self, final_report: Dict):
        """打印综合报告"""
        print(f"\n{'='*70}")
        print(f"模型评估综合报告")
        print(f"{'='*70}")
        
        ranking = final_report['ranking']
        summary = final_report['summary']
        
        print(f"\n🏆 模型排名:")
        print(f"{'排名':<4} {'模型':<25} {'综合得分':<8} {'基础能力':<8} {'智能能力':<8} {'等级':<6}")
        print("-" * 70)
        
        for item in ranking:
            print(f"{item['rank']:<4} {item['model']:<25} {item['score']:<8.3f} "
                  f"{item['basic_score']:<8.3f} {item['intelligent_score']:<8.3f} {item['level']:<6}")
        
        print(f"\n总结:")
        print(f"  • 最佳模型: {summary['best_model']} (得分: {summary['best_score']:.3f})")
        print(f"  • 平均得分: {summary['average_score']:.3f}")
        print(f"  • 优秀模型: {summary['excellent_count']} 个")
        print(f"  • 评估总数: {summary['total_models']} 个")
    
    def _create_fallback_evaluation(self, model_name: str) -> Dict[str, Any]:
        """创建回退评估结果"""
        return {
            'model_name': model_name,
            'basic_capability': {'total_basic_score': 0.5},
            'intelligent_capability': {'total_intelligent_score': 0.35},
            'final_score': 0.44,
            'performance_level': '一般',
            'analysis': {
                'summary': f"{model_name} 评估过程中出现错误",
                'strengths': [],
                'improvements': ['评估稳定性需要提升'],
                'recommendations': ['检查模型输出格式一致性']
            }
        }

# 添加快捷函数
def create_layered_evaluator(config_path: str = None) -> LayeredEvaluator:
    """创建分层评估器实例"""
    return LayeredEvaluator(config_path)