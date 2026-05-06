from milvus import default_server

print("Starting Milvus Lite server...")
# 启动本地 Milvus Lite 服务器，并指定持久化数据目录
default_server.set_base_dir("milvus_data")
default_server.start()

print(f"Milvus server is running at: {default_server.listen_port}")
