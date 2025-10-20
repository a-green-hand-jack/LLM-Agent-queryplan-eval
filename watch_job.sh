#!/bin/bash

JOB_ID=${1:-41260834}
LOG_DIR="logs"

echo "监控任务: $JOB_ID"
echo "================================"
echo ""

# 循环检查任务状态
while true; do
    clear
    echo "任务状态: $(date)"
    squeue -j $JOB_ID
    
    # 列出日志文件
    if ls $LOG_DIR/test_local_model_${JOB_ID}.out 2>/dev/null; then
        echo ""
        echo "最新日志 (最后 50 行):"
        echo "================================"
        tail -50 $LOG_DIR/test_local_model_${JOB_ID}.out
    fi
    
    echo ""
    echo "按 Ctrl+C 退出，或等待自动刷新..."
    sleep 10
done
