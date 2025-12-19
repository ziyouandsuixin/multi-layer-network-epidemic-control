import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime
from evaluation import create_layered_evaluator

class LayeredComparativeExperiment:
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.layered_evaluator = create_layered_evaluator()
        self.experiment_config = self._load_experiment_config()
        self.results = {}
        self.comparison_data = {}
        
    def _load_experiment_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.config_dir, "experiment_config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"加载实验配置失败: {e}")
                
        return {
            'time_steps': 30,
            'population_size': 500,
            'initial_infected_ratio': 0.05,
            'output_dir': 'outputs/comparative_experiments'
        }
    
    def run_comparative_experiment(self, models: List[Any], network_data: Any, 
                                 initial_states: Dict[str, Any]) -> Dict[str, Any]:
        
        print("\n" + "="*60)
        print("开始分层对比实验")
        print("="*60)
        
        self.results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            model_results = self._run_models_simulation(models, network_data, initial_states)
            evaluation_results = self._evaluate_models_layered(models, model_results)
            comparison_report = self._generate_comparison_report(evaluation_results)
            self._save_results(evaluation_results, comparison_report, timestamp)
            
            print(f"分层对比实验完成!")
            print(f"模型排名: {comparison_report['ranking']}")
            
            return {
                'evaluation_results': evaluation_results,
                'comparison_report': comparison_report,
                'success': True
            }
            
        except Exception as e:
            print(f"对比实验失败: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def _run_models_simulation(self, models: List[Any], network_data: Any, 
                             initial_states: Dict[str, Any]) -> Dict[str, Any]:
        model_results = {}
        
        for model in models:
            print(f"--- 运行模型: {model.name} ---")
            try:
                results = model.simulate(network_data, initial_states, 
                                       self.experiment_config['time_steps'])
                model_results[model.name] = results
                print(f"{model.name} 模拟完成")
                
            except Exception as e:
                print(f"{model.name} 模拟失败: {e}")
                model_results[model.name] = self._create_fallback_results()
                
        return model_results
    
    def _evaluate_models_layered(self, models: List[Any], 
                               model_results: Dict[str, Any]) -> Dict[str, Any]:
        evaluation_results = {}
        
        for model in models:
            model_name = model.name
            results = model_results.get(model_name, {})
            
            print(f"评估模型: {model_name}")
            
            simulation_logs = self._get_simulation_logs(model, results)
            
            layered_evaluation = self.layered_evaluator.evaluate_model(
                model_name, results, simulation_logs
            )
            
            evaluation_results[model_name] = layered_evaluation
            
        return evaluation_results
    
    def _get_simulation_logs(self, model: Any, results: Dict[str, Any]) -> str:
        logs = ""
        
        if hasattr(model, 'simulation_logs'):
            logs = getattr(model, 'simulation_logs', "")
            
        elif 'simulation_logs' in results:
            logs = results['simulation_logs']
            
        elif 'log_file' in results:
            log_file = results['log_file']
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        logs = f.read()
                except:
                    pass
                    
        return logs
    
    def _generate_comparison_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        print(f"生成对比分析报告...")
        
        rankings = self._calculate_rankings(evaluation_results)
        strengths_analysis = self._analyze_model_strengths(evaluation_results)
        recommendations = self._generate_recommendations(evaluation_results)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'ranking': rankings,
            'strengths_analysis': strengths_analysis,
            'recommendations': recommendations,
            'summary': self._generate_summary(rankings, strengths_analysis)
        }
        
        return report
    
    def _calculate_rankings(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        scores = []
        
        for model_name, results in evaluation_results.items():
            final_score = results.get('final_score', 0)
            scores.append({
                'model': model_name,
                'score': final_score,
                'basic_score': results['layers']['basic_capability'].get('total_basic_score', 0),
                'intelligent_score': results['layers']['intelligent_capability'].get('total_intelligent_score', 0)
            })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        for i, item in enumerate(scores):
            item['rank'] = i + 1
            
        return scores
    
    def _analyze_model_strengths(self, evaluation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        strengths = {}
        
        for model_name, results in evaluation_results.items():
            model_strengths = []
            
            basic_scores = results['layers']['basic_capability']
            intelligent_scores = results['layers']['intelligent_capability']
            
            if basic_scores.get('epidemic_control', 0) > 0.7:
                model_strengths.append("疫情控制")
            if basic_scores.get('resource_efficiency', 0) > 0.7:
                model_strengths.append("资源效率")
            if basic_scores.get('socioeconomic', 0) > 0.7:
                model_strengths.append("社会经济")
                
            if intelligent_scores.get('risk_prediction', 0) > 0.7:
                model_strengths.append("风险预测")
            if intelligent_scores.get('targeted_intervention', 0) > 0.7:
                model_strengths.append("精准干预")
            if intelligent_scores.get('adaptability', 0) > 0.7:
                model_strengths.append("适应性")
                
            strengths[model_name] = model_strengths
            
        return strengths
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        recommendations = {}
        
        for model_name, results in evaluation_results.items():
            analysis = results.get('analysis', {})
            recs = analysis.get('recommendations', [])
            recommendations[model_name] = recs
            
        return recommendations
    
    def _generate_summary(self, rankings: List[Dict], strengths: Dict[str, List[str]]) -> str:
        if not rankings:
            return "无有效评估结果"
            
        best_model = rankings[0]
        best_strengths = strengths.get(best_model['model'], [])
        
        summary = f"最佳模型: {best_model['model']} (得分: {best_model['score']:.3f})"
        if best_strengths:
            summary += f"，优势领域: {', '.join(best_strengths)}"
            
        return summary
    
    def _save_results(self, evaluation_results: Dict[str, Any], 
                     comparison_report: Dict[str, Any], timestamp: str):
        output_dir = self.experiment_config.get('output_dir', 'outputs/comparative_experiments')
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_file = os.path.join(output_dir, f"layered_results_{timestamp}.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'evaluation_results': evaluation_results,
                'comparison_report': comparison_report,
                'experiment_config': self.experiment_config
            }, f, ensure_ascii=False, indent=2)
        
        summary_file = os.path.join(output_dir, f"summary_report_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存:")
        print(f"  详细结果: {detailed_file}")
        print(f"  摘要报告: {summary_file}")
    
    def _create_fallback_results(self) -> Dict[str, Any]:
        return {
            'peak_infection': 0,
            'total_cases': 0,
            'resource_usage': {},
            'constraints': {},
            'lockdown_ratio': 0,
            'economic_cost': 0,
            'success': False
        }
    
    def print_final_ranking(self, comparison_report: Dict[str, Any]):
        rankings = comparison_report.get('ranking', [])
        strengths = comparison_report.get('strengths_analysis', {})
        
        print(f"\n{'='*60}")
        print("最终模型排名")
        print(f"{'='*60}")
        
        for i, model in enumerate(rankings):
            rank_icon = ["1", "2", "3", "4", "5", "6"][i] if i < 6 else str(i+1)
            model_strengths = strengths.get(model['model'], [])
            
            print(f"{rank_icon} {model['model']}: {model['score']:.3f}")
            print(f"   基础能力: {model['basic_score']:.3f} | 智能能力: {model['intelligent_score']:.3f}")
            if model_strengths:
                print(f"   优势: {', '.join(model_strengths)}")
            print()

def run_layered_comparative_experiment(models, network_data, initial_states, config_dir="config"):
    experiment = LayeredComparativeExperiment(config_dir)
    return experiment.run_comparative_experiment(models, network_data, initial_states)
