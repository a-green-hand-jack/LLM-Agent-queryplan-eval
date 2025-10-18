#!/usr/bin/env python3
"""
通用可视化工具类

基于visualization_style.yaml配置文件的可视化工具，
提供一致的画图风格和便捷的绘图函数。

使用方法:
    from visualization_utils import VisualizationManager
    
    # 初始化
    viz = VisualizationManager("config/visualization_style.yaml")
    
    # 使用预设风格绘图
    viz.create_scatter_plot(x_data, y_data, "scatter_plot.png")

作者: AI Assistant
日期: 2025-09-23
"""

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings

class VisualizationManager:
    """可视化管理器类"""
    
    def __init__(self, config_path: str = "config/visualization_style.yaml"):
        """初始化可视化管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._setup_matplotlib()
        self._setup_seaborn()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'global_settings': {
                'style': 'whitegrid',
                'font_family': 'DejaVu Sans',
                'figure_size': [10, 8],
                'dpi': 300,
                'alpha': 0.7,
                'grid_alpha': 0.3
            },
            'font_sizes': {
                'title': 16,
                'axis_label': 14,
                'tick_label': 12,
                'legend': 12
            },
            'colors': {
                'strategy_colors': {
                    'no_template': '#2E8B57',
                    'full_template': '#4682B4',
                    'mask_all_cdr': '#CD853F'
                },
                'trend_line': '#FF6347'
            }
        }
    
    def _setup_matplotlib(self) -> None:
        """设置matplotlib参数"""
        global_settings = self.config.get('global_settings', {})
        font_sizes = self.config.get('font_sizes', {})
        
        # 字体设置
        plt.rcParams['font.sans-serif'] = [global_settings.get('font_family', 'DejaVu Sans')]
        plt.rcParams['axes.unicode_minus'] = False
        
        # 图表尺寸和分辨率
        plt.rcParams['figure.figsize'] = global_settings.get('figure_size', [10, 8])
        plt.rcParams['figure.dpi'] = global_settings.get('dpi', 300)
        
        # 字体大小
        plt.rcParams['font.size'] = font_sizes.get('tick_label', 12)
        plt.rcParams['axes.titlesize'] = font_sizes.get('title', 16)
        plt.rcParams['axes.labelsize'] = font_sizes.get('axis_label', 14)
        plt.rcParams['legend.fontsize'] = font_sizes.get('legend', 12)
    
    def _setup_seaborn(self) -> None:
        """设置seaborn样式"""
        style = self.config.get('global_settings', {}).get('style', 'whitegrid')
        sns.set_style(style)
    
    def get_colors(self) -> Dict[str, str]:
        """获取颜色配置"""
        return self.config.get('colors', {}).get('strategy_colors', {})
    
    def create_scatter_plot(self, x_data: np.ndarray, y_data: np.ndarray, 
                           groups: Optional[np.ndarray] = None,
                           group_names: Optional[List[str]] = None,
                           output_path: str = "scatter_plot.png",
                           title: str = "Scatter Plot",
                           xlabel: str = "X Variable",
                           ylabel: str = "Y Variable",
                           add_trend_line: bool = True) -> None:
        """创建散点图
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            groups: 分组数据（可选）
            group_names: 分组名称（可选）
            output_path: 输出路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            add_trend_line: 是否添加趋势线
        """
        fig, ax = plt.subplots(figsize=self.config['global_settings']['figure_size'])
        
        colors = self.get_colors()
        scatter_config = self.config.get('scatter_plot', {})
        
        if groups is not None and group_names is not None:
            # 分组散点图
            for i, group_name in enumerate(group_names):
                mask = groups == group_name
                color = colors.get(group_name, f'C{i}')
                ax.scatter(x_data[mask], y_data[mask], 
                          c=color, 
                          label=group_name.replace('_', ' ').title(),
                          alpha=scatter_config.get('marker_alpha', 0.7),
                          s=scatter_config.get('marker_size', 60))
        else:
            # 简单散点图
            ax.scatter(x_data, y_data,
                      alpha=scatter_config.get('marker_alpha', 0.7),
                      s=scatter_config.get('marker_size', 60))
        
        # 添加趋势线
        if add_trend_line:
            self._add_trend_line(ax, x_data, y_data)
        
        # 设置标签和标题
        ax.set_xlabel(xlabel, fontsize=self.config['font_sizes']['axis_label'], fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=self.config['font_sizes']['axis_label'], fontweight='bold')
        ax.set_title(title, fontsize=self.config['font_sizes']['title'], fontweight='bold')
        
        # 设置网格和图例
        ax.grid(True, alpha=self.config['global_settings']['grid_alpha'])
        if groups is not None:
            ax.legend(loc='best', fontsize=self.config['font_sizes']['legend'])
        
        # 保存图片
        self._save_figure(fig, output_path)
        plt.close()
    
    def create_box_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                       output_path: str = "box_plot.png",
                       title: str = "Box Plot",
                       xlabel: str = "Category",
                       ylabel: str = "Value") -> None:
        """创建箱线图
        
        Args:
            data: 数据DataFrame
            x_col: X轴列名（分类变量）
            y_col: Y轴列名（数值变量）
            output_path: 输出路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
        """
        fig, ax = plt.subplots(figsize=self.config['global_settings']['figure_size'])
        
        colors = self.get_colors()
        box_config = self.config.get('box_plot', {})
        
        # 获取分类和数据
        categories = data[x_col].unique()
        box_data = [data[data[x_col] == cat][y_col].values for cat in categories]
        
        # 创建箱线图
        bp = ax.boxplot(box_data, 
                       labels=[cat.replace('_', ' ').title() for cat in categories],
                       patch_artist=True,
                       notch=box_config.get('notch', True))
        
        # 设置颜色
        for patch, category in zip(bp['boxes'], categories):
            patch.set_facecolor(colors.get(category, '#333333'))
            patch.set_alpha(box_config.get('patch_alpha', 0.7))
        
        # 设置标签和标题
        ax.set_xlabel(xlabel, fontsize=self.config['font_sizes']['axis_label'], fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=self.config['font_sizes']['axis_label'], fontweight='bold')
        ax.set_title(title, fontsize=self.config['font_sizes']['title'], fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=self.config['global_settings']['grid_alpha'])
        
        # 旋转X轴标签
        ax.tick_params(axis='x', rotation=self.config.get('axes', {}).get('x_rotation', 45))
        
        # 保存图片
        self._save_figure(fig, output_path)
        plt.close()
    
    def create_heatmap(self, data: pd.DataFrame, 
                      output_path: str = "heatmap.png",
                      title: str = "Heatmap") -> None:
        """创建热图
        
        Args:
            data: 数据DataFrame
            output_path: 输出路径
            title: 图表标题
        """
        heatmap_config = self.config.get('heatmap', {})
        colors_config = self.config.get('colors', {})
        
        fig, ax = plt.subplots(figsize=self.config.get('chart_types', {}).get('correlation_heatmap', {}).get('figure_size', [8, 6]))
        
        sns.heatmap(data, 
                   annot=heatmap_config.get('annot', True),
                   cmap=colors_config.get('heatmap_colormap', 'RdBu_r'),
                   center=colors_config.get('heatmap_center', 0),
                   square=heatmap_config.get('square', True),
                   linewidths=heatmap_config.get('linewidths', 0.5),
                   cbar_kws={"shrink": heatmap_config.get('cbar_shrink', 0.8)},
                   fmt=heatmap_config.get('fmt', '.3f'),
                   ax=ax)
        
        ax.set_title(title, fontsize=self.config['font_sizes']['title'], fontweight='bold')
        
        # 保存图片
        self._save_figure(fig, output_path)
        plt.close()
    
    def create_histogram(self, data: np.ndarray, 
                        groups: Optional[np.ndarray] = None,
                        group_names: Optional[List[str]] = None,
                        output_path: str = "histogram.png",
                        title: str = "Histogram",
                        xlabel: str = "Value",
                        ylabel: str = "Frequency") -> None:
        """创建直方图
        
        Args:
            data: 数据数组
            groups: 分组数据（可选）
            group_names: 分组名称（可选）
            output_path: 输出路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
        """
        fig, ax = plt.subplots(figsize=self.config['global_settings']['figure_size'])
        
        colors = self.get_colors()
        hist_config = self.config.get('histogram', {})
        
        if groups is not None and group_names is not None:
            # 分组直方图
            for i, group_name in enumerate(group_names):
                mask = groups == group_name
                color = colors.get(group_name, f'C{i}')
                ax.hist(data[mask], 
                       alpha=hist_config.get('alpha', 0.6),
                       label=group_name.replace('_', ' ').title(),
                       color=color,
                       bins=hist_config.get('bins', 20),
                       edgecolor=hist_config.get('edge_color', 'black'),
                       linewidth=hist_config.get('edge_width', 0.5))
        else:
            # 简单直方图
            ax.hist(data,
                   alpha=hist_config.get('alpha', 0.6),
                   bins=hist_config.get('bins', 20),
                   edgecolor=hist_config.get('edge_color', 'black'),
                   linewidth=hist_config.get('edge_width', 0.5))
        
        # 设置标签和标题
        ax.set_xlabel(xlabel, fontsize=self.config['font_sizes']['axis_label'], fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=self.config['font_sizes']['axis_label'], fontweight='bold')
        ax.set_title(title, fontsize=self.config['font_sizes']['title'], fontweight='bold')
        
        # 设置网格和图例
        ax.grid(True, alpha=self.config['global_settings']['grid_alpha'])
        if groups is not None:
            ax.legend(loc='best', fontsize=self.config['font_sizes']['legend'])
        
        # 保存图片
        self._save_figure(fig, output_path)
        plt.close()
    
    def _add_trend_line(self, ax, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """添加趋势线"""
        trend_config = self.config.get('scatter_plot', {}).get('trend_line', {})
        
        # 计算趋势线
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        
        # 绘制趋势线
        ax.plot(x_data, p(x_data), 
               color=trend_config.get('color', '#FF6347'),
               linestyle=trend_config.get('style', '--'),
               alpha=trend_config.get('alpha', 0.8),
               linewidth=trend_config.get('width', 2),
               label='Trend Line')
    
    def _save_figure(self, fig, output_path: str) -> None:
        """保存图片"""
        output_config = self.config.get('output', {})
        save_params = output_config.get('save_params', {})
        
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图片
        fig.savefig(output_path,
                   dpi=save_params.get('dpi', 300),
                   bbox_inches=save_params.get('bbox_inches', 'tight'),
                   facecolor=save_params.get('facecolor', 'white'),
                   edgecolor=save_params.get('edgecolor', 'none'),
                   transparent=save_params.get('transparent', False))
        
        print(f"Figure saved to: {output_path}")

# 便捷函数
def load_visualization_style(config_path: str = "config/visualization_style.yaml") -> VisualizationManager:
    """加载可视化风格
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        VisualizationManager实例
    """
    return VisualizationManager(config_path)

def quick_scatter(x_data, y_data, output_path: str = "scatter.png", **kwargs):
    """快速散点图"""
    viz = VisualizationManager()
    viz.create_scatter_plot(x_data, y_data, output_path=output_path, **kwargs)

def quick_boxplot(data: pd.DataFrame, x_col: str, y_col: str, 
                 output_path: str = "boxplot.png", **kwargs):
    """快速箱线图"""
    viz = VisualizationManager()
    viz.create_box_plot(data, x_col, y_col, output_path=output_path, **kwargs)

def quick_heatmap(data: pd.DataFrame, output_path: str = "heatmap.png", **kwargs):
    """快速热图"""
    viz = VisualizationManager()
    viz.create_heatmap(data, output_path=output_path, **kwargs)

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    
    # 使用可视化管理器
    viz = VisualizationManager()
    viz.create_scatter_plot(x, y, 
                           output_path="example_scatter.png",
                           title="Example Scatter Plot",
                           xlabel="X Variable",
                           ylabel="Y Variable")
    
    print("Example visualization created!")
