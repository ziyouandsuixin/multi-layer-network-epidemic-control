"""
增强版实验运行器 - 修复序列化版本
用于运行对比实验并处理序列化问题
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

class ExperimentRunner:
    """增强版实验运行器 - 修复序列化问题"""
    
    def __init__(self, output_dir: str = "outputs/comparative_experiments"):
        self.output_dir = output_dir
        self.models = {}
        self.experiment_results = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def register_model(self, name: str, model):
        """注册模型到实验运行器"""
        self.models[name] = model
    
    def run_comprehensive_experiment(self, 
                                   network_data: Any,
                                   initial_conditions: Dict[str, Any],
                                   available_resources: Dict[str, float],
                                   time_steps: int = 30,
                                   random_seed: int = 42) -> Dict[str, Any]:
        """运行综合对比实验"""
        
        # 修复：确保numpy随机种子设置
        np.random.seed(random_seed)
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # 运行流行病模拟
                epidemic_results = model.simulate(
                    network_data=network_data,
                    initial_states=initial_conditions,
                    time_steps=time_steps
                )
                
                # 运行资源分配（带约束）
                risk_assessment = epidemic_results.get('risk_assessment', {})
                ideal_allocation = model.allocate_resources(risk_assessment, available_resources)
                
                results[model_name] = {
                    'success': True,
                    'epidemic_results': epidemic_results,
                    'ideal_allocation': ideal_allocation,
                    'initial_conditions': initial_conditions,
                    'resource_usage': {
                        'available_resources': available_resources
                    }
                }
                
            except Exception as e:
                results[model_name] = {
                    'success': False,
                    'error': str(e),
                    'epidemic_results': {},
                    'ideal_allocation': {},
                }
        
        # 综合评估
        from .enhanced_evaluator import EnhancedEvaluator
        evaluator = EnhancedEvaluator()
        evaluation_report = evaluator.generate_comparative_report(results)
        
        # 保存完整结果
        self.experiment_results = {
            'results': results,
            'evaluation': evaluation_report,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'time_steps': time_steps,
                'random_seed': random_seed,
                'initial_conditions': initial_conditions,
                'available_resources': available_resources
            }
        }
        
        self._save_complete_results()
        
        return self.experiment_results
    
    def _save_complete_results(self):
        """保存完整实验结果 - 修复序列化问题"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_file = f"detailed_results_{timestamp}.json"
        detailed_path = os.path.join(self.output_dir, detailed_file)
        
        def default_serializer(obj):
            """自定义序列化器 - 处理各种不可序列化的对象"""
            try:
                # 处理numpy类型
                if isinstance(obj, (np.integer, np.int_)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float_)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                
                # 处理有__dict__的对象
                elif hasattr(obj, '__dict__'):
                    serialized_dict = {}
                    for key, value in obj.__dict__.items():
                        if not key.startswith('_'):
                            try:
                                serialized_dict[key] = default_serializer(value)
                            except (TypeError, ValueError):
                                serialized_dict[key] = str(value)
                    return serialized_dict
                
                # 处理其他可迭代对象
                elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                    return [default_serializer(item) for item in obj]
                
                # 处理其他类型
                else:
                    return str(obj)
                    
            except Exception as e:
                return f"<Unserializable: {type(obj).__name__}>"
        
        try:
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_results, f, indent=2, ensure_ascii=False, default=default_serializer)
        except Exception as e:
            # 尝试保存简化版本
            self._save_simplified_results()
        
        # 保存摘要报告
        summary_file = f"summary_report_{timestamp}.json"
        summary_path = os.path.join(self.output_dir, summary_file)
        
        summary = {
            'ranking': self.experiment_results['evaluation']['ranking'] if 'evaluation' in self.experiment_results else [],
            'timestamp': self.experiment_results['metadata']['timestamp'] if 'metadata' in self.experiment_results else datetime.now().isoformat(),
            'successful_models': len([r for r in self.experiment_results.get('results', {}).values() if r.get('success', False)]),
            'total_models': len(self.experiment_results.get('results', {}))
        }
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass
    
    def _save_simplified_results(self):
        """保存简化版结果 - 作为备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simplified_file = f"simplified_results_{timestamp}.json"
        simplified_path = os.path.join(self.output_dir, simplified_file)
        
        # 创建简化版本，只包含关键信息
        simplified_results = {
            'ranking': self.experiment_results.get('evaluation', {}).get('ranking', []),
            'model_scores': {},
            'metadata': self.experiment_results.get('metadata', {})
        }
        
        # 提取各模型得分
        comparative_scores = self.experiment_results.get('evaluation', {}).get('comparative_scores', {})
        for model_name, scores in comparative_scores.items():
            simplified_results['model_scores'][model_name] = {
                'comprehensive_score': scores.get('comprehensive_score', 0),
                'epidemic_control': scores.get('epidemic_control', 0),
                'resource_efficiency': scores.get('resource_efficiency', 0),
                'socioeconomic_impact': scores.get('socioeconomic_impact', 0)
            }
        
        try:
            with open(simplified_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass
    
    def get_registered_models(self) -> List[str]:
        """获取已注册模型列表"""
        return list(self.models.keys())
    
    def clear_models(self):
        """清空已注册模型"""
        self.models.clear()