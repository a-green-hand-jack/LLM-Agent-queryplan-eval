"""计划数据集模块 - 支持多工作表与动态场景映射

支持从 plan_data.xlsx 加载四个工作表：
- single: query, date, domain, sub, is_personal, time, food, type
- multi: history, query, date, domain, sub, is_personal, time, food, time_frame
- single_think: query, date, domain, sub, is_personal, time, food, think
- multi_think: history, query, date, domain, sub, is_personal, time, food, think

并动态加载"健康场景"和"运动场景"作为权威映射表。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PlanDataItem:
    """计划数据项 - 通用基类"""
    idx: int
    mode: str  # 'single', 'multi', 'single_think', 'multi_think'
    query: str
    date: str
    domain: Optional[str]
    sub: Optional[str]
    is_personal: Optional[bool]
    time: Optional[str]
    food: Optional[str]
    
    # 模式特定字段
    type: Optional[str] = None  # single mode
    time_frame: Optional[str] = None  # multi mode
    history: Optional[str] = None  # multi, multi_think mode
    think: Optional[str] = None  # single_think, multi_think mode


class PlanDataset:
    """计划数据集加载器
    
    从 Excel 文件加载四个工作表，并读取健康/运动场景说明表作为权威枚举映射。
    
    Example:
        >>> dataset = PlanDataset("data/plan_data.xlsx", mode="single")
        >>> len(dataset)
        100
        >>> item = dataset[0]
        >>> print(item.query, item.domain)
        >>> # 获取权威枚举
        >>> domains = dataset.get_domains()
        >>> subs = dataset.get_subs_map()
    """
    
    def __init__(self, xlsx_path: str, mode: str = "single", n: Optional[int] = None) -> None:
        """初始化数据集
        
        Args:
            xlsx_path: Excel 文件路径
            mode: 工作表模式 ('single', 'multi', 'single_think', 'multi_think')
            n: 返回行数。如果为 None 则返回全部；如果指定，使用固定随机种子采样
        
        Raises:
            FileNotFoundError: 如果 Excel 文件不存在
            ValueError: 如果工作表或列不存在
        """
        xlsx_path_obj = Path(xlsx_path)
        if not xlsx_path_obj.exists():
            raise FileNotFoundError(f"Excel 文件不存在: {xlsx_path}")
        
        if mode not in ("single", "multi", "single_think", "multi_think"):
            raise ValueError(f"mode 必须是以下之一: single, multi, single_think, multi_think，得到: {mode}")
        
        self.xlsx_path = xlsx_path_obj
        self.mode = mode
        self._domains: Optional[Dict[str, List[str]]] = None
        self._subs_map: Optional[Dict[str, List[str]]] = None
        
        # 加载场景枚举映射
        self._load_scenario_maps()
        
        # 加载对应工作表的数据
        self._df = self._load_worksheet(mode, n)
        
        logger.info(f"从 {xlsx_path} 加载了工作表 '{mode}'，共 {len(self._df)} 行")
    
    def _load_scenario_maps(self) -> None:
        """加载健康场景和运动场景作为权威映射表"""
        try:
            # 读取"健康场景"表
            health_df = pd.read_excel(self.xlsx_path, sheet_name="健康场景")
            health_domains = health_df.iloc[:, 0].dropna().tolist()
            logger.info(f"从健康场景表加载 {len(health_domains)} 个健康领域")
            
            # 读取"运动场景"表
            sports_df = pd.read_excel(self.xlsx_path, sheet_name="运动场景")
            sports_domains = sports_df.iloc[:, 0].dropna().tolist()
            logger.info(f"从运动场景表加载 {len(sports_domains)} 个运动领域")
            
            # 构建权威枚举字典
            self._domains = {
                "健康领域": health_domains,
                "运动领域": sports_domains,
                "其他领域": ["其他"]
            }
            
            # 构建 sub 映射（若二级表存在则读取）
            # 这里简化为按运动领域的子场景（如有则读取，否则置空）
            self._subs_map = self._extract_subs_map()
            
        except Exception as e:
            logger.warning(f"加载场景映射表失败: {e}，将使用默认枚举")
            # 回退到默认枚举
            self._domains = {
                "健康领域": ["体温", "减脂", "心脏健康", "情绪健康", "生理健康", "血压", "血氧饱和度", "血糖", "睡眠", "午睡", "步数", "活力三环", "微体检", "饮食"],
                "运动领域": ["跑步", "骑行", "步行徒步", "游泳", "登山", "跳绳", "瑜伽", "普拉提", "划船机", "椭圆机", "高尔夫", "潜水", "自由训练", "电子竞技", "跳操", "核心训练", "射箭", "赛车", "跳舞", "飞盘", "攀岩", "CrossFit", "钓鱼", "体能训练", "足球", "轮滑", "爬楼", "拳击", "冲浪", "滑雪", "太极拳", "乒乓球", "功能性训练", "力量训练", "网球", "自由搏击", "篮球", "滑板", "HIIT", "排球", "羽毛球", "漫步机", "踏步机", "团体操", "跆拳道", "单杠", "双杠", "芭蕾", "武术", "呼啦圈", "拔河", "台球", "保龄球", "毽球", "BMX", "冬季两项", "冰壶", "冰球", "击剑", "剑道", "垒球", "壁球", "定向越野", "对战游戏", "帆船", "手球", "摩托艇", "放风筝", "曲棍球", "板球", "棒球", "橄榄球", "沙滩排球", "沙滩足球", "浆板冲浪", "滑冰", "漂流", "皮划艇", "秋千", "空手道", "笼式网球", "藤球", "赛艇", "越野滑雪", "跑酷", "跳伞", "蹦极", "躲避球", "铁人三项", "门球", "障碍赛", "雪板滑雪", "雪橇", "雪车", "骑马", "龙舟", "户外探险", "所有运动"],
                "其他领域": ["其他"]
            }
            self._subs_map = {}
    
    def _extract_subs_map(self) -> Dict[str, List[str]]:
        """从表格中提取 sub 映射（若有二级结构）
        
        当前实现简化版：仅返回空映射，可后续扩展为支持二级表结构。
        """
        # TODO: 若 Excel 中包含子场景的二级表，此处可扩展解析逻辑
        return {
            "跑步": ["户外跑步", "室内跑步", "越野跑"],
            "步行徒步": ["户外步行", "室内步行", "徒步"],
            "骑行": ["户外骑行", "室内骑行", "动感单车"],
            "游泳": ["泳池游泳", "开放水域游泳"],
            "高尔夫": ["高尔夫场地", "高尔夫练习场"],
            "跳操": ["搏击操", "健身操"],
            "跳舞": ["肚皮舞", "广场舞", "街舞", "爵士舞", "拉丁舞", "其他舞蹈"]
        }
    
    def _load_worksheet(self, mode: str, n: Optional[int]) -> pd.DataFrame:
        """加载指定工作表的数据
        
        Args:
            mode: 工作表名称
            n: 采样数量
        
        Returns:
            规范化后的 DataFrame
        """
        try:
            df = pd.read_excel(self.xlsx_path, sheet_name=mode)
        except Exception as e:
            raise ValueError(f"无法读取工作表 '{mode}': {e}")
        
        # 规范化列名（小写、去空格）
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        
        # 定义每个模式需要的列
        required_cols = {
            "single": {"query", "date", "domain", "sub", "is_personal", "time", "food", "type"},
            "multi": {"history", "query", "date", "domain", "sub", "is_personal", "time", "food", "time_frame"},
            "single_think": {"query", "date", "domain", "sub", "is_personal", "time", "food", "think"},
            "multi_think": {"history", "query", "date", "domain", "sub", "is_personal", "time", "food", "think"}
        }
        
        needed = required_cols[mode]
        missing = needed - set(cols)
        if missing:
            raise ValueError(
                f"工作表 '{mode}' 缺少必需的列: {missing}。"
                f"可用列: {df.columns.tolist()}"
            )
        
        # 采样或使用全部数据
        if n is not None:
            df = df.sample(n=n, random_state=42)
        
        return df.reset_index(drop=True)
    
    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self._df)
    
    def __getitem__(self, idx: int) -> PlanDataItem:
        """获取指定索引的数据项
        
        Args:
            idx: 数据项的索引（0 到 len-1）
        
        Returns:
            PlanDataItem 对象
        
        Raises:
            IndexError: 如果索引超出范围
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self) - 1}]")
        
        row = self._df.iloc[idx]
        
        # 构建数据项（根据模式选择字段）
        item_dict = {
            "idx": idx,
            "mode": self.mode,
            "query": str(row.get("query", "")).strip(),
            "date": str(row.get("date", "")).strip(),
            "domain": self._safe_str(row.get("domain")),
            "sub": self._safe_str(row.get("sub")),
            "is_personal": self._safe_bool(row.get("is_personal")),
            "time": self._safe_str(row.get("time")),
            "food": self._safe_str(row.get("food")),
            "type": self._safe_str(row.get("type")) if self.mode in ("single",) else None,
            "time_frame": self._safe_str(row.get("time_frame")) if self.mode in ("multi",) else None,
            "history": self._safe_str(row.get("history")) if self.mode in ("multi", "multi_think") else None,
            "think": self._safe_str(row.get("think")) if self.mode in ("single_think", "multi_think") else None,
        }
        
        return PlanDataItem(**item_dict)
    
    def __iter__(self):
        """迭代数据集中的所有项"""
        for i in range(len(self)):
            yield self[i]
    
    def get_domains(self) -> Dict[str, List[str]]:
        """获取权威 domains 映射表"""
        return self._domains or {}
    
    def get_subs_map(self) -> Dict[str, List[str]]:
        """获取权威 sub 映射表"""
        return self._subs_map or {}
    
    @staticmethod
    def _safe_str(val: Any) -> Optional[str]:
        """安全地转换为字符串，处理 NaN/None"""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        return s if s else None
    
    @staticmethod
    def _safe_bool(val: Any) -> Optional[bool]:
        """安全地转换为布尔值"""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes", "是")
        return bool(val)
