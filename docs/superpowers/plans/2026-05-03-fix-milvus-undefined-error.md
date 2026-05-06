# Milvus undefined undefined Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Next.js 中 Milvus 初始化的 `undefined undefined` 报错，阻止回退到 MemoryVectorStore，确保 RAG 系统能成功连接真实 Milvus 数据库。

**Architecture:**
底层根本原因已定位为 `protobufjs` 在 `Node 22` 环境下处理 gRPC 序列化时触发的 `ERR_BUFFER_OUT_OF_BOUNDS` 越界异常。由于 `@zilliz/milvus2-sdk-node` 依赖 `@grpc/grpc-js` 和 `protobufjs`，这个底层报错在经过多层 try-catch 后丢失了原始堆栈，被包装成了幽灵报错 `undefined undefined`。
同时，Next.js 的 Turbopack 环境尝试去打包这些包含了 C++ 扩展/原生缓冲操作的库，导致内存布局错误被进一步放大。

为了彻底根治，我们需要：
1. 使用 `.nvmrc` 锁定当前工程的运行时环境为 `v20.x`，避免开发者系统全局的 `Node 22` 环境污染 Next.js 进程。
2. 修改 `next.config.ts`，利用 `serverExternalPackages` 将 `Milvus` 及其底层 `gRPC` 依赖排除出 Turbopack 的打包范围，让它们以原生的 Node.js 模块方式运行。

**Tech Stack:** Next.js (Turbopack), Node.js (v20), @zilliz/milvus2-sdk-node

---

### Task 1: 锁定 Node.js 运行版本

**Files:**
- Create: `.nvmrc`
- Modify: `package.json`

- [ ] **Step 1: 创建 .nvmrc 文件**

```bash
echo "20" > .nvmrc
```

- [ ] **Step 2: 修改 package.json 增加 engines 限制**

在 `package.json` 的顶层增加 `engines` 字段，确保 npm/yarn 在安装和执行时给出警告。

```json
  "engines": {
    "node": ">=20.0.0 <22.0.0"
  },
```

- [ ] **Step 3: Commit**

```bash
git add .nvmrc package.json
git commit -m "chore: lock node version to v20 to prevent protobufjs buffer out of bounds error in node v22"
```

### Task 2: 配置 Next.js 服务端外部依赖

**Files:**
- Modify: `next.config.ts`

- [ ] **Step 1: 添加 serverExternalPackages 配置**

将 Milvus 和 gRPC 相关的包标记为外部依赖，防止 Turbopack 打包导致的原型链断裂和内存越界。

修改 `next.config.ts`：

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  serverExternalPackages: [
    "@zilliz/milvus2-sdk-node",
    "@grpc/grpc-js",
    "@grpc/proto-loader",
    "protobufjs"
  ],
};

export default nextConfig;
```

- [ ] **Step 2: Commit**

```bash
git add next.config.ts
git commit -m "fix: exclude milvus and grpc dependencies from next.js server build"
```

### Task 3: 验证修复效果

**Files:**
- Test: 本地运行并验证接口

- [ ] **Step 1: 重启所有服务**

```bash
# 先停掉所有相关进程
npm run db:down
pkill -f 'next/dist/bin/next dev' || true

# 重新启动数据库和开发服务器（确保在 Node 20 环境下）
npm run db:up
npm run dev
```

- [ ] **Step 2: 发送验证请求**

等待 `npm run dev` 启动成功后，在另一个终端执行：

```bash
curl -sS 'http://127.0.0.1:3000/api/documents?page=1&pageSize=5'
```

- [ ] **Step 3: 观察终端日志**

确认终端不再输出 `Error: undefined undefined` 或 `ERR_BUFFER_OUT_OF_BOUNDS`，且不再出现 `Falling back to MemoryVectorStore`，即可宣布修复成功。
