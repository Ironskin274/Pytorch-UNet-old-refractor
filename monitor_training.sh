#!/bin/bash
# 监控训练脚本（包含wandb链接）

echo "======================================"
echo "BraTS2020训练监控"
echo "======================================"
echo ""

# 读取进程ID
if [ -f logs/train_pid.txt ]; then
    PID=$(cat logs/train_pid.txt)
    echo "训练进程ID: ${PID}"
    
    # 检查进程是否还在运行
    if ps -p ${PID} > /dev/null; then
        echo "✓ 训练进程正在运行"
        echo ""
        
        # 获取最新日志文件
        LOG_FILE=$(ls -t logs/train_brats_*.log 2>/dev/null | head -n1)
        
        if [ -n "${LOG_FILE}" ]; then
            # 提取wandb链接
            WANDB_URL=$(grep -o 'https://wandb.ai[^[:space:]]*' ${LOG_FILE} | tail -n1)
            if [ -n "${WANDB_URL}" ]; then
                echo "📊 Weights & Biases Dashboard:"
                echo "${WANDB_URL}"
                echo ""
            fi
            
            # 显示最新的Dice分数
            echo "最新Dice分数:"
            grep "Validation Dice" ${LOG_FILE} | tail -n 5
            echo ""
            
            # 显示GPU使用情况
            echo "GPU使用情况:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
            echo ""
            
            # 显示进程运行时间
            echo "进程运行时间:"
            ps -p ${PID} -o etime=
            echo ""
            
            # 显示最新日志（最后30行）
            echo "最新日志（最后30行）:"
            echo "--------------------------------------"
            tail -n 30 ${LOG_FILE}
            echo ""
            echo "--------------------------------------"
            echo "完整日志: ${LOG_FILE}"
            echo "实时查看: tail -f ${LOG_FILE}"
        else
            echo "未找到日志文件"
        fi
    else
        echo "✗ 训练进程已停止"
        echo ""
        LOG_FILE=$(ls -t logs/train_brats_*.log 2>/dev/null | head -n1)
        if [ -n "${LOG_FILE}" ]; then
            echo "查看日志了解详情:"
            echo "  tail -n 100 ${LOG_FILE}"
            echo ""
            # 显示最后的Dice分数
            echo "最终Dice分数:"
            grep "Validation Dice" ${LOG_FILE} | tail -n 5
        fi
    fi
else
    echo "未找到训练进程信息"
    echo "请先运行 ./train_brats_nohup.sh 启动训练"
fi

echo ""
echo "======================================"

