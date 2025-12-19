# utils/scientific_sampler.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class ScientificSampler:
    """基于流行病学原理的科学抽样器"""
    
    def __init__(self, target_size=500, random_state=42):
        self.target_size = target_size
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
    def sample_mobility_data(self, mobility_data, case_data=None):
        """对移动数据进行科学抽样"""
        if len(mobility_data) <= self.target_size:
            return mobility_data
        
        print(f"对移动数据执行科学抽样: {len(mobility_data)} → {self.target_size}")
        
        weights = self._calculate_sampling_weights(mobility_data, case_data)
        
        unique_individuals = mobility_data['individual_id'].unique()
        sampled_individuals = self._weighted_sample_individuals(
            unique_individuals, weights, self.target_size
        )
        
        sampled_data = mobility_data[
            mobility_data['individual_id'].isin(sampled_individuals)
        ].copy()
        
        print(f"抽样完成: {len(sampled_individuals)} 个个体, {len(sampled_data)} 条记录")
        return sampled_data
    
    def sample_exposure_data(self, exposure_data, sampled_individuals):
        """对暴露数据进行科学抽样"""
        if len(exposure_data) <= self.target_size:
            return exposure_data
        
        print(f"对暴露数据执行科学抽样: {len(exposure_data)} → {self.target_size}")
        
        sampled_exposure = exposure_data[
            exposure_data['case_id'].isin(sampled_individuals)
        ].copy()
        
        if len(sampled_exposure) > self.target_size:
            risk_weights = {
                'high': 3.0, 'medium': 1.5, 'low': 1.0
            }
            sampled_exposure['risk_weight'] = sampled_exposure['risk_level'].map(
                lambda x: risk_weights.get(x, 1.0)
            )
            sampled_exposure = sampled_exposure.sample(
                n=self.target_size, 
                weights='risk_weight',
                random_state=self.random_state
            )
        
        print(f"暴露数据抽样完成: {len(sampled_exposure)} 条记录")
        return sampled_exposure
    
    def _calculate_sampling_weights(self, mobility_data, case_data):
        """计算基于流行病学风险的抽样权重"""
        weights = {}
        
        individual_groups = mobility_data.groupby('individual_id')
        
        for individual_id, group in individual_groups:
            weight = 1.0
            
            movement_count = len(group)
            weight *= min(movement_count / 10, 3.0)
            
            unique_locations = group['location_id'].nunique()
            weight *= min(unique_locations / 5, 2.0)
            
            if 'date' in group.columns:
                early_movement_bonus = self._calculate_early_movement_bonus(group)
                weight *= early_movement_bonus
            
            if case_data is not None:
                case_weight = self._get_case_status_weight(individual_id, case_data)
                weight *= case_weight
            
            weights[individual_id] = weight
        
        return weights
    
    def _calculate_early_movement_bonus(self, group):
        """计算早期移动奖励权重"""
        try:
            dates = pd.to_datetime(
                group['date'], 
                errors='coerce',
                format='%m/%d/%Y'
            )
            valid_dates = dates[dates.notna()]
            if len(valid_dates) == 0:
                return 1.0
            
            earliest_date = valid_dates.min()
            reference_date = pd.to_datetime('2020-02-01')
            
            if earliest_date < reference_date:
                days_early = (reference_date - earliest_date).days
                return 1.0 + min(days_early / 30, 2.0)
        except Exception as e:
            self.logger.debug(f"日期解析失败: {e}")
        
        return 1.0
    
    def _get_case_status_weight(self, individual_id, case_data):
        """获取病例状态权重"""
        if individual_id in case_data['case_id'].values:
            case_info = case_data[case_data['case_id'] == individual_id].iloc[0]
            
            severity_weights = {
                'severe': 2.5,
                'moderate': 1.8, 
                'mild': 1.2
            }
            
            severity = case_info.get('severity', 'mild')
            return severity_weights.get(severity, 1.0)
        
        return 1.0
    
    def _weighted_sample_individuals(self, individuals, weights, n):
        """执行加权抽样"""
        individual_list = list(individuals)
        weight_list = [weights.get(ind, 1.0) for ind in individual_list]
        
        total_weight = sum(weight_list)
        normalized_weights = [w / total_weight for w in weight_list]
        
        sampled_indices = np.random.choice(
            len(individual_list), 
            size=min(n, len(individual_list)),
            replace=False,
            p=normalized_weights
        )
        
        return [individual_list[i] for i in sampled_indices]