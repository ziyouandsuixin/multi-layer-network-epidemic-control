#!/usr/bin/env python3
"""
数据提取器测试脚本
测试从模拟日志文件中提取结构化数据
"""
import os
import re
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class CommunitySnapshot:
    """社区在某个时间步的快照"""
    time_step: int
    community_id: str
    community_size: int
    infected_count: int
    risk_score: float
    infection_pressure: float
    has_super_spreader: bool

@dataclass
class CommunityTimeline:
    """社区的时间序列数据"""
    community_id: str
    snapshots: List[CommunitySnapshot]

class SimulationDataExtractor:
    """模拟数据提取器"""
    
    def __init__(self):
        self.community_timelines: Dict[str, CommunityTimeline] = {}
        self.global_infection_data: List[Dict] = []
    
    def extract_from_logs(self, log_text: str) -> None:
        """
        从模拟日志文本中提取数据 - 更精确的版本
        """
        lines = log_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 直接为每个社区分析提取时间步
            if "时间步" in line and "社区" in line:
                # 这种格式: "时间步 0 - 社区 0_0: 31/560 感染"
                time_match = re.search(r'\s*时间步\s*(\d+)\s*-\s*社区', line)
                if time_match:
                    current_time_step = int(time_match.group(1))
                    
                    # 找到对应的社区风险分析（通常在后面几行）
                    for j in range(i+1, min(i+10, len(lines))):
                        if "社区风险分析" in lines[j]:
                            community_data = self._extract_community_risk_analysis(
                                lines[j:j+10], current_time_step
                            )
                            if community_data:
                                self._add_community_snapshot(community_data)
                            break
            
            # 提取全局感染数据
            elif "时间步" in line and "完成" in line:
                time_match = re.search(r'时间步\s*(\d+)\s*完成', line)
                if time_match:
                    current_time_step = int(time_match.group(1))
                    global_data = self._extract_global_infection(line, current_time_step)
                    if global_data:
                        self.global_infection_data.append(global_data)
    
    def _extract_time_step(self, line: str) -> int:
        """提取时间步编号"""
        match = re.search(r'时间步 (\d+)', line)
        return int(match.group(1)) if match else 0
    
    def _extract_community_risk_analysis(self, lines: List[str], time_step: int) -> Dict[str, Any]:
        """提取社区风险分析数据 - 修复版"""
        data = {}
        
        for line in lines:
            line = line.strip()
            
            # 提取社区ID
            if "社区风险分析" in line:
                match = re.search(r'\[ID: ([^\]]+)\]', line)
                if match:
                    data['community_id'] = match.group(1)
            
            # 提取社区大小
            elif "社区大小" in line:
                match = re.search(r'社区大小:\s*(\d+)\s*节点', line)
                if match:
                    data['community_size'] = int(match.group(1))
            
            # 提取感染情况 - 修复这里！
            elif "感染情况" in line:
                match = re.search(r'感染情况:\s*(\d+)/(\d+)\s*=\s*([\d.]+)', line)
                if match:
                    data['infected_count'] = int(match.group(1))  # 提取感染数
                    data['infection_rate'] = float(match.group(3))  # 保留感染率
            
            # 提取风险分数
            elif "总风险分数" in line:
                match = re.search(r'总风险分数:\s*([\d.]+)', line)
                if match:
                    data['risk_score'] = float(match.group(1))
            
            # 提取感染压力
            elif "感染压力:" in line:
                match = re.search(r'感染压力:\s*([\d.]+)', line)
                if match:
                    data['infection_pressure'] = float(match.group(1))
            
            # 提取超级传播者信息
            elif "超级传播者:" in line:
                data['has_super_spreader'] = "True" in line
        
        # 检查必要字段 - 现在应该都能匹配了
        required_fields = ['community_id', 'community_size', 'infected_count', 'risk_score']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"  缺少字段: {missing_fields}")
            print(f"  当前数据: {data}")
            return {}
        else:
            data['time_step'] = time_step
            print(f"  社区数据提取成功: {data}")
            return data
    
    def _extract_global_infection(self, line: str, time_step: int) -> Dict[str, Any]:
        """提取全局感染数据"""
        match = re.search(r'总感染节点数: (\d+)', line)
        if match:
            return {
                'time_step': time_step,
                'total_infected': int(match.group(1))
            }
        return {}
    
    def _add_community_snapshot(self, data: Dict[str, Any]) -> None:
        """添加社区快照数据"""
        community_id = data['community_id']
        
        snapshot = CommunitySnapshot(
            time_step=data['time_step'],
            community_id=community_id,
            community_size=data['community_size'],
            infected_count=data['infected_count'],
            risk_score=data['risk_score'],
            infection_pressure=data.get('infection_pressure', 0),
            has_super_spreader=data.get('has_super_spreader', False)
        )
        
        if community_id not in self.community_timelines:
            self.community_timelines[community_id] = CommunityTimeline(
                community_id=community_id, snapshots=[]
            )
        
        self.community_timelines[community_id].snapshots.append(snapshot)
    
    def get_community_data(self, community_id: str) -> CommunityTimeline:
        """获取指定社区的时间序列数据"""
        return self.community_timelines.get(community_id)
    
    def get_all_communities(self) -> List[str]:
        """获取所有社区ID"""
        return list(self.community_timelines.keys())
    
    def get_snapshot_at_time(self, time_step: int) -> List[CommunitySnapshot]:
        """获取指定时间步的所有社区快照"""
        snapshots = []
        for timeline in self.community_timelines.values():
            for snapshot in timeline.snapshots:
                if snapshot.time_step == time_step:
                    snapshots.append(snapshot)
        return snapshots

def test_data_extraction():
    """测试数据提取功能"""
    
    # 你的日志文件路径
    log_file_path = r"D:\whu\coursers\Social\1109\outputs\dynamics_test_20251111_145140.log"
    
    print("=== 数据提取器测试 ===")
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
        print(f"文件大小: {len(log_text)} 字符")
        
        # 创建提取器实例
        extractor = SimulationDataExtractor()
        
        # 提取数据
        print("\n开始提取数据...")
        extractor.extract_from_logs(log_text)
        
        # 显示提取结果
        print("\n=== 数据提取结果 ===")
        
        # 社区数据统计
        community_count = len(extractor.community_timelines)
        print(f"找到社区数量: {community_count}")
        
        if community_count > 0:
            print("\n各社区数据:")
            for community_id, timeline in extractor.community_timelines.items():
                print(f"\n社区 {community_id}:")
                # 按时间步排序
                sorted_snapshots = sorted(timeline.snapshots, key=lambda x: x.time_step)
                for snapshot in sorted_snapshots:
                    print(f"  时间步 {snapshot.time_step}: "
                          f"感染 {snapshot.infected_count}/{snapshot.community_size} "
                          f"(风险分数: {snapshot.risk_score:.3f})")
        
        # 全局感染数据
        print(f"\n全局感染数据:")
        for data in extractor.global_infection_data:
            print(f"  时间步 {data['time_step']}: 总感染 {data['total_infected']} 节点")
        
        # 数据质量检查
        print(f"\n数据提取完成!")
        print(f"   社区时间线: {len(extractor.community_timelines)} 条")
        print(f"   全局数据点: {len(extractor.global_infection_data)} 个")
        
        # 检查是否有足够的数据进行评估
        if community_count >= 3 and len(extractor.global_infection_data) >= 3:
            print("数据质量良好，可以进行评估")
        else:
            print("数据可能不足，请检查日志文件完整性")
            
    except Exception as e:
        print(f"数据提取失败: {e}")
        import traceback
        traceback.print_exc()

def check_data_patterns():
    """检查日志文件中的数据模式"""
    
    log_file_path = r"D:\whu\coursers\Social\1109\outputs\dynamics_test_20251111_141829.log"
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print("\n=== 数据模式检查 ===")
        
        # 统计关键模式
        time_step_starts = sum(1 for line in lines if "时间步" in line and "开始" in line)
        community_analysis = sum(1 for line in lines if "社区风险分析" in line)
        time_step_ends = sum(1 for line in lines if "时间步" in line and "完成" in line)
        total_infected = sum(1 for line in lines if "总感染节点数:" in line)
        
        print(f"时间步开始次数: {time_step_starts}")
        print(f"社区风险分析次数: {community_analysis}")
        print(f"时间步完成次数: {time_step_ends}")
        print(f"总感染节点数统计: {total_infected}")
        
        # 显示一些样本行
        print(f"\n样本数据:")
        for i, line in enumerate(lines[:20]):
            if any(keyword in line for keyword in ["时间步", "社区风险分析", "总感染节点数"]):
                print(f"  {i+1}: {line.strip()}")
                
    except Exception as e:
        print(f"模式检查失败: {e}")

def debug_data_extraction():
    """调试数据提取过程"""
    
    log_file_path = r"D:\whu\coursers\Social\1109\outputs\dynamics_test_20251111_141829.log"
    
    # 创建提取器实例
    extractor = SimulationDataExtractor()
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("=== 调试数据提取 ===")
    print(f"总行数: {len(lines)}")
    
    community_analysis_count = 0
    successful_extractions = 0
    
    for i, line in enumerate(lines):
        if "社区风险分析" in line:
            community_analysis_count += 1
            print(f"\n第 {community_analysis_count} 次社区风险分析 (第 {i+1} 行):")
            print(f"  触发行: {line.strip()}")
            
            # 显示接下来的8行内容
            context_lines = lines[i:i+8]
            print("  上下文内容:")
            for j, context_line in enumerate(context_lines):
                print(f"    {i+j+1}: {context_line.strip()}")
            
            # 测试提取函数
            test_data = extractor._extract_community_risk_analysis(context_lines, 0)
            if test_data:
                successful_extractions += 1
                print(f"  成功提取: {test_data}")
            else:
                print(f"  提取失败 - 检查正则表达式匹配")
            
            print("-" * 60)
    
    print(f"\n=== 调试总结 ===")
    print(f"检测到社区风险分析次数: {community_analysis_count}")
    print(f"成功提取次数: {successful_extractions}")
    print(f"失败次数: {community_analysis_count - successful_extractions}")

def check_specific_patterns():
    """检查具体的模式匹配"""
    
    log_file_path = r"D:\whu\coursers\Social\1109\outputs\dynamics_test_20251111_141829.log"
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("\n=== 具体模式检查 ===")
    
    # 测试各种正则表达式模式
    test_lines = []
    for i, line in enumerate(lines):
        if "社区风险分析" in line:
            # 取这个社区分析块的前8行
            test_lines = lines[i:i+8]
            break
    
    if test_lines:
        print("测试第一个社区分析块:")
        for j, line in enumerate(test_lines):
            print(f"  {j+1}: {line.strip()}")
        
        # 测试各个字段的提取
        print("\n字段提取测试:")
        
        # 社区ID
        for line in test_lines:
            if "社区风险分析" in line:
                patterns = [
                    r'\[ID: ([^\]]+)\]',
                    r'ID:\s*([^\s\]]+)',
                    r'社区风险分析\s*\[([^\]]+)\]'
                ]
                for pattern_num, pattern in enumerate(patterns):
                    match = re.search(pattern, line)
                    if match:
                        print(f"  社区ID模式{pattern_num+1}匹配: {match.group(1)}")
                    else:
                        print(f"  社区ID模式{pattern_num+1}不匹配")
        
        # 社区大小
        for line in test_lines:
            if "社区大小" in line:
                patterns = [
                    r'社区大小:\s*(\d+)\s*节点',
                    r'社区大小\s*(\d+)\s*节点'
                ]
                for pattern_num, pattern in enumerate(patterns):
                    match = re.search(pattern, line)
                    if match:
                        print(f"  社区大小模式{pattern_num+1}匹配: {match.group(1)}")
                    else:
                        print(f"  社区大小模式{pattern_num+1}不匹配")
        
        # 感染情况
        for line in test_lines:
            if "感染情况" in line:
                patterns = [
                    r'感染情况:\s*(\d+)/(\d+)\s*=\s*([\d.]+)',
                    r'感染情况:\s*(\d+)/(\d+)',
                    r'感染:\s*(\d+)/(\d+)'
                ]
                for pattern_num, pattern in enumerate(patterns):
                    match = re.search(pattern, line)
                    if match:
                        print(f"  感染情况模式{pattern_num+1}匹配: {match.group(1)}/{match.group(2)}")
                    else:
                        print(f"  感染情况模式{pattern_num+1}不匹配")
        
        # 风险分数
        for line in test_lines:
            if "总风险分数" in line:
                patterns = [
                    r'总风险分数:\s*([\d.]+)',
                    r'风险分数:\s*([\d.]+)'
                ]
                for pattern_num, pattern in enumerate(patterns):
                    match = re.search(pattern, line)
                    if match:
                        print(f"  风险分数模式{pattern_num+1}匹配: {match.group(1)}")
                    else:
                        print(f"  风险分数模式{pattern_num+1}不匹配")

if __name__ == "__main__":
    test_data_extraction()
    check_data_patterns()
    debug_data_extraction()
    check_specific_patterns()