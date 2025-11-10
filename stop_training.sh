#!/bin/bash
# 停止训练脚本

echo "======================================"
echo "停止BraTS2020训练"
echo "======================================"
echo ""

# 读取进程ID
if [ -f logs/train_pid.txt ]; then
    PID=$(cat logs/train_pid.txt)
    echo "训练进程ID: ${PID}"
    
    # 检查进程是否还在运行
    if ps -p ${PID} > /dev/null; then
        echo "正在停止训练进程..."
        kill ${PID}
        
        # 等待进程结束
        sleep 3
        
        if ps -p ${PID} > /dev/null; then
            echo "进程未响应，尝试强制停止..."
            kill -9 ${PID}
            sleep 1
        fi
        
        if ! ps -p ${PID} > /dev/null; then
            echo "✓ 训练已停止"
            echo ""
            
            # 显示最后的训练结果
            LOG_FILE=$(ls -t logs/train_brats_*.log 2>/dev/null | head -n1)
            if [ -n "${LOG_FILE}" ]; then
                echo "训练摘要:"
                echo "--------------------------------------"
                echo "最终Dice分数:"
                grep "Validation Dice" ${LOG_FILE} | tail -n 5
                echo ""
                echo "wandb链接:"
                grep -o 'https://wandb.ai[^[:space:]]*' ${LOG_FILE} | tail -n1
                echo "--------------------------------------"
            fi
        else
            echo "✗ 无法停止进程，请手动执行: kill -9 ${PID}"
        fi
    else
        echo "训练进程已经停止"
    fi
    
    # 清理PID文件
    rm -f logs/train_pid.txt
else
    echo "未找到训练进程信息"
    echo "尝试查找Python训练进程..."
    
    # 查找可能的训练进程
    PIDS=$(ps aux | grep "python.*train.py.*use-brats" | grep -v grep | awk '{print $2}')
    
    if [ -n "${PIDS}" ]; then
        echo "找到以下可能的训练进程:"
        ps aux | grep "python.*train.py.*use-brats" | grep -v grep
        echo ""
        read -p "是否停止这些进程? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            for pid in ${PIDS}; do
                echo "停止进程 ${pid}..."
                kill ${pid}
            done
            echo "✓ 已发送停止信号"
        fi
    else
        echo "未找到运行中的训练进程"
    fi
fi

echo ""
echo "======================================"

