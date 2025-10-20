"""
Span相关的工具函数

提供span的匹配、合并、overlap计算等功能
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Span定义为 (start, end) 元组,表示字符级别的位置
Span = Tuple[int, int]


def calculate_span_overlap(span1: Span, span2: Span) -> int:
    """
    计算两个span的重叠字符数
    
    Args:
        span1: 第一个span (start, end)
        span2: 第二个span (start, end)
    
    Returns:
        重叠的字符数
    
    Examples:
        >>> calculate_span_overlap((0, 5), (3, 8))
        2  # 位置3-4重叠
        >>> calculate_span_overlap((0, 5), (10, 15))
        0  # 没有重叠
    """
    start1, end1 = span1
    start2, end2 = span2
    
    # 计算重叠区间
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    # 如果没有重叠,返回0
    if overlap_start >= overlap_end:
        return 0
    
    return overlap_end - overlap_start


def merge_spans(spans: List[Span]) -> List[Span]:
    """
    合并重叠或相邻的spans
    
    Args:
        spans: span列表
    
    Returns:
        合并后的span列表
    
    Examples:
        >>> merge_spans([(0, 5), (3, 8), (10, 15)])
        [(0, 8), (10, 15)]
    """
    if not spans:
        return []
    
    # 按起始位置排序
    sorted_spans = sorted(spans, key=lambda x: x[0])
    
    merged = [sorted_spans[0]]
    
    for current in sorted_spans[1:]:
        last = merged[-1]
        
        # 如果当前span与上一个span重叠或相邻,则合并
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def parse_spans_from_text(
    hallucination_texts: List[str],
    response_text: str
) -> List[Span]:
    """
    从hallucination文本列表中解析出在response中的span位置
    
    这个函数用于将模型输出的hallucination文本片段转换为字符级别的span。
    它会在response_text中查找每个hallucination_text的所有出现位置。
    
    根据hallucination detection的标准，标点符号差异不应该影响hallucination判断，
    因此实现了模糊匹配逻辑来容忍标点差异。
    
    Args:
        hallucination_texts: 模型预测的hallucination文本列表
        response_text: 原始的response文本
    
    Returns:
        合并后的span列表，使用左闭右开区间[start, end)
    
    Examples:
        >>> texts = ["catering services", "free parking"]
        >>> response = "The restaurant provides catering services and free parking."
        >>> spans = parse_spans_from_text(texts, response)
        >>> # 返回: [(26, 44), (49, 61)]
    
    Notes:
        - 如果同一个文本在response中出现多次，会记录所有出现位置
        - 最后会合并重叠的spans
        - 如果某个hallucination_text不在response中，会尝试模糊匹配
        - 模糊匹配容忍标点符号差异（句号、逗号、引号等）
    """
    all_spans = []
    
    for hal_text in hallucination_texts:
        # 首先尝试精确匹配
        spans = _find_exact_matches(hal_text, response_text)
        if spans:
            all_spans.extend(spans)
            continue
        
        # 如果精确匹配失败，尝试模糊匹配（容忍标点差异）
        spans = _find_fuzzy_matches(hal_text, response_text)
        if spans:
            all_spans.extend(spans)
            logger.info(f"使用模糊匹配找到文本: '{hal_text}' -> {spans}")
        else:
            logger.warning(
                f"Hallucination文本 '{hal_text}' 未在response中找到（精确和模糊匹配都失败）"
            )
    
    # 排序并合并重叠的spans
    if all_spans:
        all_spans = merge_spans(all_spans)
    
    return all_spans


def _find_exact_matches(text: str, response_text: str) -> List[Span]:
    """精确匹配文本"""
    spans = []
    start = 0
    while True:
        pos = response_text.find(text, start)
        if pos == -1:
            break
        spans.append((pos, pos + len(text)))
        start = pos + 1
    return spans


def _find_fuzzy_matches(text: str, response_text: str) -> List[Span]:
    """
    模糊匹配文本，容忍标点符号差异
    
    根据hallucination detection标准，标点符号差异不应该影响判断。
    这个方法会尝试在response中找到与text语义相同但标点可能不同的文本。
    
    使用简化的方法：尝试多种标点变体进行匹配。
    """
    spans = []
    
    # 生成text的多种标点变体
    text_variants = _generate_punctuation_variants(text)
    
    # 对每个变体尝试匹配
    for variant in text_variants:
        variant_spans = _find_exact_matches(variant, response_text)
        if variant_spans:
            spans.extend(variant_spans)
            logger.info(f"模糊匹配成功: '{text}' -> '{variant}' -> {variant_spans}")
    
    return spans


def _generate_punctuation_variants(text: str) -> List[str]:
    """
    生成文本的多种标点符号变体
    
    根据hallucination detection标准，以下标点差异应该被容忍：
    - 句号、逗号、分号、冒号
    - 引号（单引号、双引号）
    - 括号
    - 连字符、破折号
    """
    variants = [text]  # 原始文本
    
    # 移除句号
    if text.endswith('.'):
        variants.append(text[:-1])
    
    # 移除逗号
    if ',' in text:
        variants.append(text.replace(',', ''))
    
    # 移除引号
    for quote in ['"', "'", '`']:
        if quote in text:
            variants.append(text.replace(quote, ''))
    
    # 移除括号
    for bracket in ['(', ')', '[', ']', '{', '}']:
        if bracket in text:
            variants.append(text.replace(bracket, ''))
    
    # 移除连字符和破折号
    for dash in ['-', '–', '—']:
        if dash in text:
            variants.append(text.replace(dash, ''))
    
    # 移除多余的空格
    import re
    normalized = re.sub(r'\s+', ' ', text).strip()
    if normalized != text:
        variants.append(normalized)
    
    # 去重
    return list(set(variants))


def spans_to_text(text: str, spans: List[Span]) -> List[str]:
    """
    根据span列表从文本中提取子字符串
    
    Args:
        text: 原始文本
        spans: span列表
    
    Returns:
        提取的文本片段列表
    
    Examples:
        >>> spans_to_text("Hello world", [(0, 5), (6, 11)])
        ['Hello', 'world']
    """
    return [text[start:end] for start, end in spans]


def spans_to_char_set(spans: List[Span]) -> set:
    """
    将span列表转换为字符位置的集合
    
    这是论文中reward计算的核心操作。每个span [start, end) 被转换为
    字符位置的集合 {start, start+1, ..., end-1}，然后所有span的字符
    集合取并集。
    
    Args:
        spans: span列表，每个span是 (start, end) 元组
               使用左闭右开区间 [start, end)
    
    Returns:
        所有span覆盖的字符位置集合
    
    Examples:
        >>> spans_to_char_set([(0, 5), (7, 10)])
        {0, 1, 2, 3, 4, 7, 8, 9}
        >>> spans_to_char_set([(0, 3), (2, 5)])  # 有重叠
        {0, 1, 2, 3, 4}
    """
    char_set = set()
    for start, end in spans:
        # 使用range(start, end)得到左闭右开区间的所有位置
        char_set.update(range(start, end))
    return char_set


def calculate_span_f1(
    predicted_spans: List[Span],
    ground_truth_spans: List[Span],
) -> Tuple[float, float, float]:
    """
    计算span-level的F1分数
    
    这是论文中reward计算的核心函数。使用字符级别的集合运算:
    1. 将预测spans转为字符集合P
    2. 将真实spans转为字符集合G  
    3. Precision = |P ∩ G| / |P|
    4. Recall = |P ∩ G| / |G|
    5. F1 = 2 * Precision * Recall / (Precision + Recall)
    
    根据论文Section 2.3的定义:
    - 使用字符级集合运算,不进行span-level匹配
    - 支持span数量不匹配（预测多或少都能正确计算）
    - 支持span长度不匹配和部分重叠
    - 重叠的spans会自动去重（使用集合的并集）
    
    Args:
        predicted_spans: 预测的span列表，使用左闭右开区间 [start, end)
        ground_truth_spans: 真实的span列表，使用左闭右开区间 [start, end)
    
    Returns:
        (precision, recall, f1) 三元组
    
    Notes:
        - 如果预测和真实都为空,F1定义为1.0
        - spans使用左闭右开区间 [start, end)，与Python标准一致
        - 部分匹配也会得到相应的分数
        - 不需要担心span数量或长度不匹配,集合运算自动处理
    
    Examples:
        >>> calculate_span_f1([(0, 10)], [(0, 10)])
        (1.0, 1.0, 1.0)
        >>> calculate_span_f1([(0, 10)], [(5, 15)])  # 50%重叠
        (0.5, 0.5, 0.5)
        >>> calculate_span_f1([(0, 5), (10, 15)], [(0, 15)])  # 数量不匹配但F1可计算
        (1.0, 0.6666..., 0.8)
    """
    # 特殊情况:都为空时F1为1
    if not predicted_spans and not ground_truth_spans:
        return 1.0, 1.0, 1.0
    
    # 一个为空另一个不为空,F1为0
    if not predicted_spans or not ground_truth_spans:
        return 0.0, 0.0, 0.0
    
    # 转换为字符集合
    P = spans_to_char_set(predicted_spans)
    G = spans_to_char_set(ground_truth_spans)
    
    # 计算交集
    intersection = P & G
    intersection_size = len(intersection)
    
    # 计算precision和recall（不使用epsilon，因为已经处理了空集情况）
    precision = intersection_size / len(P)
    recall = intersection_size / len(G)
    
    # 计算F1（如果precision和recall都为0，F1自然为0）
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

