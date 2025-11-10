#!/bin/bash
# BraTS2020后台训练脚本（使用nohup + wandb）

echo "======================================"
echo "BraTS2020 训练启动脚本"
echo "======================================"

# 创建日志目录
mkdir -p logs

# 生成时间戳作为日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_brats_${TIMESTAMP}.log"

echo ""
echo "开始训练..."
echo "日志文件: ${LOG_FILE}"
echo "wandb项目: BraTS2020-UNet"
echo "wandb: 将使用您已登录的账号"
echo ""

# 使用nohup后台运行训练
nohup python -u train.py \
    --use-brats \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --scale 1.0 \
    --amp \
    --bilinear \
    > ${LOG_FILE} 2>&1 &

# 获取进程ID
PID=$!
echo ${PID} > logs/train_pid.txt

echo "======================================"
echo "✓ 训练已启动！"
echo "======================================"
echo ""
echo "进程ID: ${PID}"
echo "日志文件: ${LOG_FILE}"
echo ""
echo "等待5秒获取wandb链接..."
sleep 5

# 尝试从日志中提取wandb链接
WANDB_URL=$(grep -o 'https://wandb.ai[^[:space:]]*' ${LOG_FILE} | tail -n1)
if [ -n "${WANDB_URL}" ]; then
    echo "======================================"
    echo "📊 Weights & Biases Dashboard:"
    echo "${WANDB_URL}"
    echo "======================================"
    echo ""
    echo "复制上面的链接到浏览器查看实时训练进度！"
else
    echo "正在等待wandb初始化..."
    echo "请稍后查看日志获取wandb链接："
    echo "  grep 'wandb.ai' ${LOG_FILE}"
fi

echo ""
echo "======================================"
echo "常用命令："
echo "======================================"
echo "查看实时日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "查看wandb链接:"
echo "  grep 'wandb.ai' ${LOG_FILE}"
echo ""
echo "监控训练状态:"
echo "  ./monitor_training.sh"
echo ""
echo "查看GPU使用:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "停止训练:"
echo "  ./stop_training.sh"
echo "  或: kill ${PID}"
echo "======================================"

