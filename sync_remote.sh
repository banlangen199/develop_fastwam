#!/bin/bash
# ============================================
# auto_sync_from_remote.sh
# 自动从远程服务器拉取指定目录到本地
# ============================================

#关于rsync的源文件路径，最后不带 /，复制整个文件夹（包括它自己）；带 /，只复制文件夹内部内容，不包含文件夹本身
# === 固定配置 ===
REMOTE_IP="218.200.126.238"        # 远程服务器IP
REMOTE_USER="txc"                   # 登录用户名
SSH_PORT="39118"                   # SSH端口
REMOTE_PATH="/mnt/data/txc/FastWAM/data/libero_mujoco3.3.2"  # 要拉取的远程目录
LOCAL_PATH="/mnt/hwdata/txc/FastWAM/data/libero_mujoco3.3.2"                    # 本地保存路径
RSYNC_OPTS="-avuz --progress"       # rsync参数
LOG_DIR="/mnt/hwdata/txc/tool/rsync_log"   # 日志目录

# === 新增：是否包含整个文件夹 ===
# 如果 INCLUDE_DIR=true，则复制整个文件夹（包括自身）
# 如果 INCLUDE_DIR=false，则只复制文件夹内的内容
INCLUDE_DIR=true

# === 新增：是否排除某些子文件夹 ===
# 如果 EXCLUDE_DIRS 为空，则不排除任何子文件夹
# 这里填写的是相对于 REMOTE_PATH 的子文件夹路径
# 例如：
#   "checkpoints" 会排除 REMOTE_PATH/checkpoints/
#   "logs"        会排除 REMOTE_PATH/logs/
#   "runs/debug"  会排除 REMOTE_PATH/runs/debug/
#
# 注意：
# rsync 的 --exclude="logs/" 会排除所有名字叫 logs 的目录
# 如果只想排除某个更具体的目录，建议写成类似 "experiments/logs"
EXCLUDE_DIRS=(
    weights
)

# === 新增：让管道命令能正确返回 rsync 的失败状态 ===
# 否则 rsync ... | tee -a logfile 后，$? 可能拿到的是 tee 的状态
set -o pipefail

# === 初始化日志 ===
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/sync_log_$(date +%F_%H-%M-%S).log"

echo "==============================================="
echo "📥 开始从远程服务器拉取文件夹..."
echo "  来源:   ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"
echo "  目标:   ${LOCAL_PATH}"
echo "  端口:   ${SSH_PORT}"
echo "  模式:   $( [ "$INCLUDE_DIR" = true ] && echo "包含文件夹本身" || echo "仅文件夹内容" )"
echo "  日志文件: ${LOGFILE}"
echo "  排除目录: ${EXCLUDE_DIRS[*]:-无}"
echo "==============================================="

# 检查连接
nc -z -w3 "$REMOTE_IP" "$SSH_PORT"
if [ $? -ne 0 ]; then
    echo "❌ 无法连接到远程服务器 ${REMOTE_IP}:${SSH_PORT}"
    exit 1
fi

# === 新增：生成 rsync 的排除参数 ===
EXCLUDE_OPTS=()
for dir in "${EXCLUDE_DIRS[@]}"; do
    EXCLUDE_OPTS+=(--exclude="${dir}/")
done

# 根据模式选择同步方式
if [ "$INCLUDE_DIR" = true ]; then
    # 同步整个文件夹本身（不加 /）
    RSYNC_CMD=(
        rsync
        $RSYNC_OPTS
        "${EXCLUDE_OPTS[@]}"
        -e "ssh -p $SSH_PORT"
        "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"
        "${LOCAL_PATH}"
    )
else
    # 仅同步文件夹内容（加 /）
    RSYNC_CMD=(
        rsync
        $RSYNC_OPTS
        "${EXCLUDE_OPTS[@]}"
        -e "ssh -p $SSH_PORT"
        "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}/"
        "${LOCAL_PATH}"
    )
fi

echo -n "执行命令："
printf "%q " "${RSYNC_CMD[@]}"
echo

"${RSYNC_CMD[@]}" | tee -a "$LOGFILE"

if [ $? -eq 0 ]; then
    echo "✅ 同步完成！" | tee -a "$LOGFILE"
else
    echo "⚠️ 同步过程中出现错误，请查看日志: ${LOGFILE}" | tee -a "$LOGFILE"
fi

echo "==============================================="