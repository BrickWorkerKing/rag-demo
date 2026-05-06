#!/bin/bash

echo "=== 开始清理 Milvus 相关进程和残留文件 ==="

# 1. 查找并终止进程
echo "1. 查找并终止 Milvus 相关进程..."
# 查找 milvus, etcd, minio 进程（排除 grep 自身）
PIDS=$(ps aux | grep -i -E "milvus|etcd|minio" | grep -v grep | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "找到以下进程 PID: $PIDS，正在强制终止..."
    kill -9 $PIDS
    echo "✅ 进程已终止。"
else
    echo "✅ 未发现运行中的 Milvus 相关进程。"
fi

# 2. 清理 /tmp/milvus 临时文件
echo "2. 清理 /tmp/milvus 临时文件..."
if [ -d "/tmp/milvus" ]; then
    rm -rf /tmp/milvus/*
    echo "✅ /tmp/milvus 下的 PID 等临时文件已清理。"
else
    echo "✅ /tmp/milvus 目录不存在，无需清理。"
fi

# 3. 清理 RocksMQ 锁文件
echo "3. 清理数据库锁文件..."
LOCK_FILES=(
    "milvus_data/data/rocksmq/LOCK"
    "milvus_data/data/rocksmq_meta_kv/LOCK"
)

for file in "${LOCK_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "✅ 已删除锁文件: $file"
    fi
done

echo "=== 清理完成！现在你可以尝试重新启动 Milvus 了。 ==="
