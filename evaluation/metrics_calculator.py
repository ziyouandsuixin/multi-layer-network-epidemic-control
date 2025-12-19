"""
评估指标计算器
"""
import numpy as np
from typing import List, Dict, Any
from .data_extractor import CommunitySnapshot, CommunityTimeline

class CommunityRiskMetrics:
    """社区风险相关指标计算"""
    
    @staticmethod
    def calculate_ndcg(predicted_rank: List[str], actual_rank: List[str]) -> float:
        """
        计算NDCG (归一化折损累积增益)
        
        Args:
            predicted_rank: 预测的风险排名（社区ID列表，从高到低）
            actual_rank: 实际的风险排名（社区ID列表，从高到低）
        
        Returns:
            NDCG分数 (0-1之间)
        """
        def dcg(rank_list: List[str], relevance_scores: Dict[str, float]) -> float:
            """计算折损累积增益"""
            score = 0.0
            for i, community_id in enumerate(rank_list):
                relevance = relevance_scores.get(community_id, 0)
                discount = np.log2(i + 2)  # i+2因为排名从0开始，但log要从2开始
                score += relevance / discount
            return score
        
        # 构建相关性分数：排名越靠前，分数越高
        relevance_scores = {}
        for i, community_id in enumerate(actual_rank):
            relevance_scores[community_id] = len(actual_rank) - i
        
        # 计算预测排名的DCG
        pred_dcg = dcg(predicted_rank, relevance_scores)
        
        # 计算理想排名的DCG
        ideal_dcg = dcg(actual_rank, relevance_scores)
        
        return pred_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def calculate_risk_prediction_accuracy(
        timelines: Dict[str, CommunityTimeline],
        prediction_time: int,
        evaluation_period: int = 2
    ) -> Dict[str, Any]:
        """
        计算社区风险预测准确度
        
        Args:
            timelines: 社区时间序列数据
            prediction_time: 进行预测的时间步
            evaluation_period: 评估的时间跨度
        
        Returns:
            包含NDCG和其他指标的字典
        """
        # 获取预测时刻的社区快照
        prediction_snapshots = []
        for timeline in timelines.values():
            for snapshot in timeline.snapshots:
                if snapshot.time_step == prediction_time:
                    prediction_snapshots.append(snapshot)
                    break
        
        if not prediction_snapshots:
            return {"error": f"在时间步 {prediction_time} 没有找到数据"}
        
        # 按风险分数排序（预测排名）
        predicted_rank = sorted(
            prediction_snapshots, 
            key=lambda x: x.risk_score, 
            reverse=True
        )
        predicted_community_ids = [s.community_id for s in predicted_rank]
        
        # 计算实际感染增长率并排序（实际排名）
        actual_growth_rates = {}
        for snapshot in prediction_snapshots:
            community_id = snapshot.community_id
            timeline = timelines[community_id]
            
            # 找到预测时刻和评估结束时刻的感染数
            start_infected = snapshot.infected_count
            end_infected = start_infected
            
            for later_snapshot in timeline.snapshots:
                if (later_snapshot.time_step > prediction_time and 
                    later_snapshot.time_step <= prediction_time + evaluation_period):
                    end_infected = later_snapshot.infected_count
            
            # 计算感染增长率
            growth_rate = (end_infected - start_infected) / snapshot.community_size
            actual_growth_rates[community_id] = growth_rate
        
        # 按实际增长率排序
        actual_rank = sorted(
            actual_growth_rates.keys(),
            key=lambda x: actual_growth_rates[x],
            reverse=True
        )
        
        # 计算NDCG
        ndcg_score = CommunityRiskMetrics.calculate_ndcg(
            predicted_community_ids, actual_rank
        )
        
        return {
            "prediction_time": prediction_time,
            "evaluation_period": evaluation_period,
            "ndcg_score": ndcg_score,
            "predicted_rank": predicted_community_ids,
            "actual_rank": actual_rank,
            "actual_growth_rates": actual_growth_rates
        }

class InfectionMetrics:
    """感染动力学指标计算"""
    
    @staticmethod
    def calculate_peak_infection(global_data: List[Dict]) -> float:
        """计算峰值感染率"""
        if not global_data:
            return 0.0
        max_infected = max(entry['total_infected'] for entry in global_data)
        # 假设总节点数从第一个社区快照获取
        total_nodes = 700  # 可以从数据中动态获取
        return max_infected / total_nodes
    
    @staticmethod
    def calculate_final_infection_rate(global_data: List[Dict]) -> float:
        """计算最终感染率"""
        if not global_data:
            return 0.0
        final_infected = global_data[-1]['total_infected']
        total_nodes = 700
        return final_infected / total_nodes