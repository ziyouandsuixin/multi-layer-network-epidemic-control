"""
评估主入口 - 统一接口
"""
from typing import Dict, Any
from .data_extractor import SimulationDataExtractor
from .metrics_calculator import CommunityRiskMetrics, InfectionMetrics

class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        self.extractor = SimulationDataExtractor()
        self.metrics_results = {}
    
    def _get_available_time_steps(self, timelines):
        """获取所有可用的时间步"""
        available_times = set()
        for timeline in timelines.values():
            for snapshot in timeline.snapshots:
                available_times.add(snapshot.time_step)
        return sorted(available_times)
    
    def _find_optimal_prediction_time(self, timelines, min_evaluation_period=1):
        """
        寻找最优的预测时间点
        
        Args:
            timelines: 社区时间序列数据
            min_evaluation_period: 最小评估周期
            
        Returns:
            prediction_time: 预测时间点
            evaluation_period: 评估周期
        """
        available_times = self._get_available_time_steps(timelines)
        
        if not available_times:
            return None, 0
        
        # 选择最早的时间点作为预测时间
        prediction_time = available_times[0]
        
        # 计算可用的评估周期
        max_time = max(available_times)
        evaluation_period = max_time - prediction_time
        
        # 确保评估周期至少为 min_evaluation_period
        if evaluation_period < min_evaluation_period:
            # 如果没有足够的数据，返回None表示无法评估
            return None, 0
        
        return prediction_time, min(evaluation_period, 3)  # 限制最大评估周期为3
    
    def evaluate_simulation(self, simulation_log: str) -> Dict[str, Any]:
        """
        评估一次模拟运行
        
        Args:
            simulation_log: 模拟输出日志
            
        Returns:
            包含所有评估指标的字典
        """
        # 提取数据
        self.extractor.extract_from_logs(simulation_log)
        
        results = {}
        
        # 计算社区风险预测准确度
        timelines = self.extractor.community_timelines
        if timelines:
            # 动态确定预测时间和评估周期
            prediction_time, evaluation_period = self._find_optimal_prediction_time(
                timelines, min_evaluation_period=1
            )
            
            if prediction_time is not None:
                risk_metrics = CommunityRiskMetrics.calculate_risk_prediction_accuracy(
                    timelines, 
                    prediction_time=prediction_time, 
                    evaluation_period=evaluation_period
                )
                results['community_risk_accuracy'] = risk_metrics
                # 存储使用的参数供报告使用
                results['prediction_params'] = {
                    'prediction_time': prediction_time,
                    'evaluation_period': evaluation_period,
                    'available_time_steps': self._get_available_time_steps(timelines)
                }
            else:
                results['community_risk_accuracy'] = {
                    "error": "数据不足，无法进行风险预测评估",
                    "available_time_steps": self._get_available_time_steps(timelines)
                }
        
        # 计算感染动力学指标（保持不变）
        if self.extractor.global_infection_data:
            results['peak_infection_rate'] = InfectionMetrics.calculate_peak_infection(
                self.extractor.global_infection_data
            )
            results['final_infection_rate'] = InfectionMetrics.calculate_final_infection_rate(
                self.extractor.global_infection_data
            )
        
        # 存储结果
        self.metrics_results = results
        return results
    
    def generate_report(self) -> str:
        """生成评估报告"""
        report = ["=== 模拟评估报告 ==="]
        
        # 显示可用的时间步信息
        if 'prediction_params' in self.metrics_results:
            params = self.metrics_results['prediction_params']
            report.append(f"可用时间步: {params['available_time_steps']}")
            report.append(f"使用的预测时间: {params['prediction_time']}")
            report.append(f"使用的评估周期: {params['evaluation_period']}")
        
        if 'community_risk_accuracy' in self.metrics_results:
            risk_data = self.metrics_results['community_risk_accuracy']
            
            # 错误检查
            if 'error' in risk_data:
                report.append(f"社区风险预测评估失败: {risk_data['error']}")
            elif 'ndcg_score' in risk_data:
                report.append(f"社区风险预测准确度 (NDCG): {risk_data['ndcg_score']:.3f}")
                report.append(f"预测时间步: {risk_data['prediction_time']}")
                report.append(f"评估周期: {risk_data['evaluation_period']} 步")
                
                # 显示排名对比
                report.append("\n排名对比:")
                report.append("预测排名 vs 实际排名")
                for i, (pred, actual) in enumerate(zip(risk_data['predicted_rank'], risk_data['actual_rank'])):
                    pred_growth = risk_data['actual_growth_rates'].get(pred, 0)
                    actual_growth = risk_data['actual_growth_rates'].get(actual, 0)
                    report.append(f"  {i+1}. {pred} (增长率: {pred_growth:.3f}) | {actual} (增长率: {actual_growth:.3f})")
            else:
                report.append("社区风险预测数据不完整")
        
        if 'peak_infection_rate' in self.metrics_results:
            report.append(f"\n峰值感染率: {self.metrics_results['peak_infection_rate']:.3f}")
        
        if 'final_infection_rate' in self.metrics_results:
            report.append(f"最终感染率: {self.metrics_results['final_infection_rate']:.3f}")
        
        return "\n".join(report)