#!/usr/bin/env python3
"""
测试指标计算
"""
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 现在可以正确导入
from evaluation.evaluator import ComprehensiveEvaluator
from evaluation.metrics_calculator import CommunityRiskMetrics

def test_metrics_calculation():
    """测试指标计算功能"""
    
    # 你的日志文件路径
    log_file_path = r"D:\whu\coursers\Social\1109\outputs\dynamics_test_20251112_140510.log"
    
    print("=== 指标计算测试 ===")
    print(f"读取文件: {log_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"文件不存在: {log_file_path}")
        return
    
    try:
        # 读取日志文件
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_text = f.read()
        
        print("文件读取成功")
        
        # 创建评估器
        evaluator = ComprehensiveEvaluator()
        
        # 执行评估
        print("\n开始计算评估指标...")
        results = evaluator.evaluate_simulation(log_text)
        
        # 生成报告
        report = evaluator.generate_report()
        print(report)
        
        # 保存详细结果
        import json
        output_dir = r"D:\whu\coursers\Social\1109\outputs"
        output_file = os.path.join(output_dir, "evaluation_results.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"指标计算失败: {e}")
        import traceback
        traceback.print_exc()

def test_different_prediction_times():
    """测试不同预测时间点的准确度"""
    
    log_file_path = r"D:\whu\coursers\Social\1109\outputs\dynamics_test_20251112_140510.log"
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()
    
    from evaluation.data_extractor import SimulationDataExtractor
    extractor = SimulationDataExtractor()
    extractor.extract_from_logs(log_text)
    
    print("\n=== 不同预测时间点测试 ===")
    
    # 测试不同预测时间点
    for prediction_time in [0, 1, 2]:
        risk_metrics = CommunityRiskMetrics.calculate_risk_prediction_accuracy(
            extractor.community_timelines,
            prediction_time=prediction_time,
            evaluation_period=2
        )
        
        if 'ndcg_score' in risk_metrics:
            print(f"预测时间步 {prediction_time}: NDCG = {risk_metrics['ndcg_score']:.3f}")
        else:
            print(f"预测时间步 {prediction_time}: {risk_metrics.get('error', '未知错误')}")

if __name__ == "__main__":
    test_metrics_calculation()
    test_different_prediction_times()