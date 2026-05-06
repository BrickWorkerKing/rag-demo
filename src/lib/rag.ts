import path from "node:path";

import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Milvus, type MilvusLibArgs } from "@langchain/community/vectorstores/milvus";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { env, pipeline } from "@xenova/transformers";

/**
 * 本地模型目录：固定指向项目根目录下的 `models`。
 *
 * 说明：
 * 1. 统一管理所有本地模型缓存。
 * 2. 避免在服务器/容器环境中把模型散落在系统临时目录。
 */
const LOCAL_MODEL_DIR = path.resolve(process.cwd(), "models");

/**
 * 全局 transformers.js 配置。
 *
 * 目标：
 * - 强制“只使用本地模型”。
 * - 禁止自动从远程下载模型，避免线上环境首次请求阻塞或失败。
 */
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = LOCAL_MODEL_DIR;

console.log(`[RAG] transformers local model dir: ${LOCAL_MODEL_DIR}`);
console.log(`[RAG] transformers allowRemoteModels: ${env.allowRemoteModels}`);

// 配置信息
// const MILVUS_CONFIG = {
//   collectionName: "rag_collection", // 向量数据库中的集合名称（类似于关系型数据库的表名）
//   clientConfig: {
//     address: process.env.MILVUS_ADDRESS || "localhost:19530", // Milvus 连接地址
//   },
// };

const DEEPSEEK_CONFIG = {
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://sg.uiuiapi.com/v1", // DeepSeek 或兼容 OpenAI 协议的 API 地址
};

/**
 * 向量库统一类型：
 * - 优先使用 Milvus（生产可持久化）
 * - 失败降级 MemoryVectorStore（开发/容错）
 */
type SupportedVectorStore = Milvus | MemoryVectorStore;

/**
 * 用于描述 transformers.js 返回的 embedding Tensor 结构。
 *
 * 典型二维形状：
 * - dims[0] = 文本条数（rows）
 * - dims[1] = 向量维度（cols）
 */
type TensorLike = {
  data: ArrayLike<number>;
  dims: number[];
};

/**
 * feature-extraction pipeline 的可调用函数类型。
 *
 * 这里显式声明是为了：
 * - 避免 `any`。
 * - 在代码中更清晰地约束输入参数和返回值。
 */
type FeatureExtractionRunner = (
  texts: string | string[],
  options?: {
    pooling?: "none" | "mean" | "cls";
    normalize?: boolean;
  }
) => Promise<TensorLike>;

/**
 * 自定义本地 Embeddings 实现。
 *
 * 设计目标：
 * 1. 直接控制 transformers.js pipeline 参数（如 cache_dir、quantized）。
 * 2. 提供 LangChain 要求的 embedDocuments/embedQuery 接口。
 * 3. 出现异常时自动回退到社区 HuggingFaceTransformersEmbeddings，提升鲁棒性。
 */
export class LocalHuggingFaceEmbeddings extends Embeddings {
  /** 使用的 embedding 模型名 */
  private readonly model: string;
  /** 模型缓存目录 */
  private readonly cacheDir: string;
  /** 批处理大小，避免一次性处理过多文本占用内存 */
  private readonly batchSize: number;

  /**
   * 延迟初始化的 pipeline Promise。
   *
   * 说明：
   * - 首次调用才真正加载模型（懒加载）。
   * - 多次并发请求复用同一个 Promise，避免重复加载模型。
   */
  private pipelinePromise: Promise<FeatureExtractionRunner> | null = null;

  /**
   * 兜底 embedding 实现。
   *
   * 仅在本地 pipeline 执行异常时使用，用于提高系统可用性。
   */
  private readonly fallbackEmbeddings: HuggingFaceTransformersEmbeddings;

  constructor(params?: {
    model?: string;
    cacheDir?: string;
    batchSize?: number;
  }) {
    super({});
    this.model = params?.model ?? "Xenova/bge-small-zh-v1.5";
    this.cacheDir = params?.cacheDir ?? LOCAL_MODEL_DIR;
    this.batchSize = params?.batchSize ?? 16;

    // 兜底 Embeddings，同样强制本地文件与相同缓存目录。
    this.fallbackEmbeddings = new HuggingFaceTransformersEmbeddings({
      model: this.model,
      batchSize: this.batchSize,
      pretrainedOptions: {
        cache_dir: this.cacheDir,
        local_files_only: true,
      },
      pipelineOptions: {
        pooling: "mean",
        normalize: true,
      },
    });
  }

  /**
   * 获取（或初始化）feature-extraction pipeline。
   *
   * 关键参数：
   * - cache_dir: 指定本地模型目录。
   * - quantized: 开启量化，减少内存占用，通常可显著降低资源压力。
   */
  private async getPipeline() {
    if (!this.pipelinePromise) {
      console.log(`[RAG] Initializing local embedding pipeline: ${this.model}`);
      this.pipelinePromise = pipeline("feature-extraction", this.model, {
        cache_dir: this.cacheDir,
        quantized: true,
      }) as Promise<FeatureExtractionRunner>;
    }

    return this.pipelinePromise;
  }

  /**
   * 把 pipeline 返回的二维 Tensor 展平成 number[][]。
   *
   * @param tensor 模型返回的张量对象
   * @param expectedRows 期望文本条数（用于一致性校验）
   */
  private toVectors(tensor: TensorLike, expectedRows: number): number[][] {
    if (!tensor?.data || !Array.isArray(tensor?.dims) || tensor.dims.length !== 2) {
      throw new Error("[RAG] Unexpected embedding tensor shape.");
    }

    const [rows, cols] = tensor.dims;
    const data = Array.from(tensor.data);
    const vectors: number[][] = [];

    for (let r = 0; r < rows; r += 1) {
      vectors.push(data.slice(r * cols, (r + 1) * cols));
    }

    // 条数不一致一般意味着上游输入/输出出现异常，先告警方便排查。
    if (rows !== expectedRows) {
      console.warn(
        `[RAG] Embedding row mismatch: expected=${expectedRows}, actual=${rows}`
      );
    }

    return vectors;
  }

  /**
   * 单批次 embedding 执行。
   *
   * 处理细节：
   * - 先把换行替换为空格，减少无意义格式差异对向量的影响。
   * - 使用 mean pooling + normalize，得到可直接用于相似度检索的向量。
   */
  private async runEmbedding(texts: string[]): Promise<number[][]> {
    const modelInput = texts.map((t) => t.replace(/\r?\n/g, " "));
    const extractor = await this.getPipeline();
    const output = (await extractor(modelInput, {
      pooling: "mean",
      normalize: true,
    })) as TensorLike;
    return this.toVectors(output, modelInput.length);
  }

  /**
   * 批量文档 embedding。
   *
   * 流程：
   * 1. 按 batchSize 切批。
   * 2. 逐批调用本地 pipeline。
   * 3. 发生异常则回退到 fallbackEmbeddings。
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    try {
      const vectors: number[][] = [];
      for (let i = 0; i < texts.length; i += this.batchSize) {
        const batch = texts.slice(i, i + this.batchSize);
        const batchVectors = await this.runEmbedding(batch);
        vectors.push(...batchVectors);
      }
      return vectors;
    } catch (error) {
      console.warn(
        "[RAG] Local embedding pipeline failed, using HuggingFace fallback.",
        error
      );
      return this.fallbackEmbeddings.embedDocuments(texts);
    }
  }

  /**
   * 单条 query embedding。
   *
   * 复用 embedDocuments 保持行为一致，最后取第一条结果。
   */
  async embedQuery(text: string): Promise<number[]> {
    const vectors = await this.embedDocuments([text]);
    if (!vectors[0]) {
      throw new Error("[RAG] Failed to generate query embedding.");
    }
    return vectors[0];
  }
}

/**
 * RAG 引擎核心类。
 *
 * 职责：
 * 1. 初始化 LLM 与 Embedding。
 * 2. 管理向量库（Milvus + 内存降级）。
 * 3. 承担文档 ETL 入库流程（加载 -> 切分 -> 向量化 -> 存储）。
 */
export class RAGEngine {
  /** 当前向量库实例，可能是 Milvus 或内存向量库 */
  public vectorStore: SupportedVectorStore | null = null;
  /** embedding 组件 */
  public readonly embeddings: LocalHuggingFaceEmbeddings;
  /** 对话模型 */
  public readonly llm: ChatOpenAI;
  /** 是否处于“内存降级模式” */
  public isMemoryStore = false;

  /**
   * init 过程的 Promise 锁。
   *
   * 作用：避免高并发下重复执行初始化逻辑。
   */
  private initPromise: Promise<void> | null = null;

  /** Milvus 连接与集合配置 */
  private readonly milvusConfig: MilvusLibArgs;

  /**
   * 统一输出当前进程与向量库状态，便于排查“命中了哪个进程、当前走了哪条分支”。
   */
  private logDebug(label: string, extra: Record<string, unknown> = {}): void {
    console.log(`[RAG Debug] ${label}`, {
      pid: process.pid,
      node: process.version,
      isMemoryStore: this.isMemoryStore,
      hasVectorStore: Boolean(this.vectorStore),
      milvusUrl: this.milvusConfig.url ?? null,
      collectionName: this.milvusConfig.collectionName ?? null,
      ...extra,
    });
  }

  /**
   * Milvus Lite 冷启动时存在一个“进程已拉起但 gRPC 尚未 ready”的短窗口。
   * 这里做一个轻量级健康检查重试，避免模块刚初始化就误判连接失败。
   */
  private async waitForMilvusHealthy(
    context: string,
    maxAttempts: number = 5,
    retryDelayMs: number = 400
  ): Promise<void> {
    const { MilvusClient } = await import("@zilliz/milvus2-sdk-node");
    let lastError: unknown = null;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      let client: InstanceType<typeof MilvusClient> | null = null;
      try {
        client = new MilvusClient({ address: this.milvusConfig.url! });
        const health = await client.checkHealth();
        this.logDebug("milvus.healthCheck", {
          context,
          attempt,
          isHealthy: health.isHealthy,
          reasons: health.reasons,
        });

        if (health.isHealthy) {
          return;
        }

        lastError = new Error("Milvus is not healthy");
      } catch (error) {
        lastError = error;
        this.logDebug("milvus.healthCheck.failed", {
          context,
          attempt,
          error,
        });
      } finally {
        if (client) {
          try {
            await client.closeConnection();
          } catch (closeError) {
            this.logDebug("milvus.healthCheck.closeFailed", {
              context,
              attempt,
              error: closeError,
            });
          }
        }
      }

      if (attempt < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
      }
    }

    throw lastError instanceof Error
      ? lastError
      : new Error(`[RAG] Milvus health check failed in ${context}.`);
  }

  constructor() {
    console.log("[RAG Constructor] 正在初始化 RAG 引擎...");
    if (!DEEPSEEK_CONFIG.apiKey) {
        throw new Error("环境变量中未设置 DEEPSEEK_API_KEY。");
    }
    if (DEEPSEEK_CONFIG.apiKey === 'mock-key') {
        console.warn("[RAG Warning] 使用 Mock API Key，LLM 功能将受限或返回模拟数据。");
    }

    // 初始化本地 embedding（中文 bge small）。
    this.embeddings = new LocalHuggingFaceEmbeddings({
      model: "Xenova/bge-small-zh-v1.5",
      cacheDir: LOCAL_MODEL_DIR,
    });

    // 初始化 ChatOpenAI，使用你指定的模型与 baseURL。
    this.llm = new ChatOpenAI({
      model: "deepseek-v3",
      apiKey: DEEPSEEK_CONFIG.apiKey, // 兼容旧命名
      configuration: {
        baseURL: DEEPSEEK_CONFIG.baseURL,
      },
      temperature: 0.7, // 随机性控制：0.7 比较平衡，既有创造性又不会太发散
    });

    // Milvus 默认连接本机 19530，可通过环境变量覆盖。
    const milvusUrl = process.env.MILVUS_URL ?? "127.0.0.1:19530";
    // 确保 url 不带 http:// 前缀，否则 gRPC 会报错 undefined undefined (retried 3 times)
    const cleanMilvusUrl = milvusUrl.replace(/^https?:\/\//, '');

    this.milvusConfig = {
      collectionName: process.env.MILVUS_COLLECTION_NAME ?? "rag_demo",
      url: cleanMilvusUrl,
      primaryField: "id",
      vectorField: "vector",
      textField: "text",
      textFieldMaxLength: 65535, // 显式声明文本字段的最大长度，防止基于第一个短文档动态推断出太短的长度（如172）导致后续追加长文档失败
      autoId: true,
      clientConfig: {
        address: cleanMilvusUrl,
      },
    };

    // 启动时记录运行时信息，方便确认请求命中的实际进程与 Milvus 连接参数。
    this.logDebug("constructor.ready", {
      rawMilvusUrl: milvusUrl,
      cleanMilvusUrl,
    });
  }

  /**
   * 初始化向量库。
   *
   * 策略：
   * 1. 先尝试连接已有 Milvus 集合（fromExistingCollection）。
   * 2. 如果失败，仅记录日志，保持 vectorStore 为 null，交由后续按需创建（如 addDocument 时）。
   */
  async init(): Promise<void> {
    if (this.initPromise) {
      this.logDebug("init.reusePromise");
      return this.initPromise;
    }

    this.initPromise = (async () => {
      this.logDebug("init.start");
      try {
        await this.waitForMilvusHealthy("init");
        
        // 1. 动态导入 MilvusClient 并检查集合是否存在
        const { MilvusClient } = await import("@zilliz/milvus2-sdk-node");
        const client = new MilvusClient({ address: this.milvusConfig.url! });
        
        const hasCollectionRes = await client.hasCollection({ 
          collection_name: this.milvusConfig.collectionName! 
        });
        
        await client.closeConnection();

        if (!hasCollectionRes.value) {
          console.log(`[RAG] Milvus collection '${this.milvusConfig.collectionName}' does not exist yet. It will be created on first document addition.`);
          this.logDebug("init.collectionNotFound");
          return;
        }

        // 2. 集合存在，进行连接
        console.log("[RAG] Trying to connect to existing Milvus collection...");
        this.vectorStore = await Milvus.fromExistingCollection(
          this.embeddings,
          this.milvusConfig
        );
        this.isMemoryStore = false;
        console.log("[RAG] Milvus connected.");
        this.logDebug("init.connected");
      } catch (error) {
        console.warn(
          "[RAG] Milvus connection failed during init.",
          error
        );
        this.logDebug("init.connectFailed", {
          error,
        });
      }
    })();

    return this.initPromise;
  }

  /**
   * 统一的降级处理逻辑：切换到内存向量库
   * @param chunks 需要初始化写入的文档块（如果有）
   */
  private async fallbackToMemoryStore(chunks: Document[] = []): Promise<void> {
    this.logDebug("fallback.memoryStore.start", {
      chunkCount: chunks.length,
    });
    this.isMemoryStore = true;
    if (chunks.length > 0) {
      this.vectorStore = await MemoryVectorStore.fromDocuments(
        chunks,
        this.embeddings
      );
      console.log(`[RAG] MemoryVectorStore created and seeded with ${chunks.length} chunks.`);
      this.logDebug("fallback.memoryStore.seeded", {
        chunkCount: chunks.length,
      });
    } else {
      this.vectorStore = new MemoryVectorStore(this.embeddings);
      console.log("[RAG] Empty MemoryVectorStore created.");
      this.logDebug("fallback.memoryStore.empty");
    }
  }

  /**
   * 按文件后缀加载原始文档。
   *
   * 支持：
   * - .csv -> CSVLoader
   * - .pdf -> PDFLoader（失败时回退 WebPDFLoader）
   * - .txt/.md -> 直接按 UTF-8 文本构造 Document
   */
  private async loadDocuments(
    fileName: string,
    fileBuffer: Buffer
  ): Promise<Document[]> {
    const ext = path.extname(fileName).toLowerCase();

    // Node Buffer 转 Blob 时使用 Uint8Array，避免 TS 在 Buffer<ArrayBufferLike> 上的兼容报错。
    const blob = new Blob([Uint8Array.from(fileBuffer)]);

    if (ext === ".csv") {
      console.log(`[RAG] Loading CSV: ${fileName}`);
      return new CSVLoader(blob).load();
    }

    if (ext === ".pdf") {
      console.log(`[RAG] Loading PDF: ${fileName}`);
      try {
        return await new PDFLoader(blob, { splitPages: true }).load();
      } catch (error) {
        // 某些环境/格式下 fs 版 PDFLoader 可能失败，回退 WebPDFLoader 提高兼容性。
        console.warn(
          "[RAG] PDFLoader failed, retrying with WebPDFLoader...",
          error
        );
        return new WebPDFLoader(blob, { splitPages: true }).load();
      }
    }

    if (ext === ".txt" || ext === ".md") {
      console.log(`[RAG] Loading text-like file: ${fileName}`);
      const text = fileBuffer.toString("utf-8");
      return [
        new Document({
          pageContent: text,
          metadata: { source: fileName, ext },
        }),
      ];
    }

    throw new Error(`[RAG] Unsupported file type: ${ext}`);
  }

  /**
   * 将原始文档进行元数据补全和切块处理
   */
  private async processDocuments(rawDocs: Document[], fileName: string): Promise<Document[]> {
    // chunkSize 按你的要求设置为 800；保留一定 overlap 提升召回连续性。
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 100,
      separators: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    });

    // 给每个文档补充统一 source，方便后续检索结果溯源。
    const docs = rawDocs.map((doc) => new Document({
      pageContent: doc.pageContent,
      metadata: { ...doc.metadata, source: fileName },
    }));

    return splitter.splitDocuments(docs);
  }

  /**
   * 将切块后的文档存入向量库
   */
  private async storeDocuments(chunks: Document[]): Promise<void> {
    this.logDebug("storeDocuments.start", {
      chunkCount: chunks.length,
    });
    // 首次入库：如果还没有 vectorStore，则先创建。
    if (!this.vectorStore) {
      if (!this.isMemoryStore) {
        try {
          console.log("[RAG] Creating Milvus collection via fromDocuments...");
          this.logDebug("storeDocuments.createMilvus.attempt", {
            chunkCount: chunks.length,
          });
          this.vectorStore = await Milvus.fromDocuments(
            chunks,
            this.embeddings,
            this.milvusConfig
          );
          console.log("[RAG] Milvus collection created and seeded.");
          this.logDebug("storeDocuments.createMilvus.success", {
            chunkCount: chunks.length,
          });
          return;
        } catch (error) {
          console.error("[RAG] Milvus create failed. Error details:", error);
          this.logDebug("storeDocuments.createMilvus.failed", {
            chunkCount: chunks.length,
            error,
          });
          console.warn("[RAG] Falling back to MemoryVectorStore.");
        }
      }
      await this.fallbackToMemoryStore(chunks);
      return;
    }

    // 非首次入库：直接追加文档。
    try {
      await this.vectorStore.addDocuments(chunks);
      console.log(`[RAG] Added ${chunks.length} chunks into vector store.`);
      this.logDebug("storeDocuments.addDocuments.success", {
        chunkCount: chunks.length,
      });
    } catch (error) {
      if (!this.isMemoryStore) {
        console.warn(
          "[RAG] Milvus addDocuments failed. Switching to MemoryVectorStore.",
          error
        );
        this.logDebug("storeDocuments.addDocuments.failed", {
          chunkCount: chunks.length,
          error,
        });
        await this.fallbackToMemoryStore(chunks);
        return;
      }
      throw error;
    }
  }

  /**
   * 文档入库主流程（ETL）。
   *
   * 完整步骤：
   * 1. init：确保向量库可用（Milvus 或内存降级）。
   * 2. load：按文件类型加载为 Document[]。
   * 3. process：切块并补充元数据。
   * 4. store：存入向量库。
   * @returns 成功入库的 chunk 数量
   */
  async addDocument(fileName: string, fileBuffer: Buffer): Promise<number> {
    this.logDebug("addDocument.start", {
      fileName,
      fileSize: fileBuffer.length,
    });
    await this.init();

    // 1、文档加载：根据文件类型加载为 Document[]
    const rawDocs = await this.loadDocuments(fileName, fileBuffer);
    if (rawDocs.length === 0) {
      console.warn("[RAG] No documents parsed, skip indexing.");
      return 0;
    }

    // 2、文档处理：切块并补充元数据
    const chunks = await this.processDocuments(rawDocs, fileName);
    console.log(
      `[RAG] Parsed ${rawDocs.length} docs from ${fileName}, split into ${chunks.length} chunks.`
    );

    // 3、元数据清洗 (Metadata Cleaning) - 对于 MemoryStore 可能不是必须的，但保留好习惯
     chunks.forEach((doc, index) => {
        // 确保 metadata 存在
        doc.metadata = doc.metadata || {};
        
        // 1. 移除已知可能导致问题的字段
        if ('blobType' in doc.metadata) {
            delete doc.metadata.blobType;
        }
        if ('loc' in doc.metadata) {
            delete doc.metadata.loc;
        }
        if ('pdf' in doc.metadata) {
            delete doc.metadata.pdf;
        }
        if ('line' in doc.metadata) {
            delete doc.metadata.line;
        }

        // 2. 扁平化处理：将所有元数据值转换为 JSON 字符串
        for (const key in doc.metadata) {
            // 确保没有 undefined/null 值
            if (doc.metadata[key] === undefined || doc.metadata[key] === null) {
                 delete doc.metadata[key];
                 continue;
            }

            const value = doc.metadata[key];
            if (typeof value === 'object' && value !== null) {
                doc.metadata[key] = JSON.stringify(value);
            }
        }
    });

    // 4、文档存储：存入向量库
    await this.storeDocuments(chunks);
    return chunks.length;
  }

  /**
   * 清空向量库中的所有数据。
   */
  async reset(): Promise<void> {
    // 1. 内存模式检查
    if (this.isMemoryStore) {
      this.vectorStore = null;
      console.log("[RAG] 内存向量库已重置。");
      return;
    }

    // 2. Milvus 模式
    let client: any = null;
    try {
      // 动态导入 @zilliz/milvus2-sdk-node 中的 MilvusClient
      const { MilvusClient } = await import("@zilliz/milvus2-sdk-node");
      
      // 创建 client 实例，直接使用整理好的 clean URL
      client = new MilvusClient({ address: this.milvusConfig.url! });

      const collectionName = this.milvusConfig.collectionName;
      if (collectionName) {
        // 删除整个集合
        await client.dropCollection({
          collection_name: collectionName,
        });
        console.log(`[RAG] Milvus collection '${collectionName}' has been dropped.`);
      }

      // 成功后将 this.vectorStore 置为 null
      this.vectorStore = null;
      this.initPromise = null;
    } catch (error) {
      // 异常处理：捕获并打印错误
      console.error("[RAG] Failed to reset Milvus collection:", error);
      throw new Error("Failed to reset knowledge base.");
    } finally {
      // 资源清理：无论成功失败，都在 finally 块中调用 client.closeConnection()
      if (client) {
        await client.closeConnection();
      }
    }
  }

  /**
   * 获取文档片段列表（支持简单的分页）。
   * 主要用于前端管理界面展示。
   *
   * @param page 页码，默认 1
   * @param pageSize 每页条数，默认 10
   */
  async getDocuments(page: number = 1, pageSize: number = 10): Promise<{ total: number, documents: Document[] }> {
    // 1. 内存模式
    if (this.isMemoryStore) {
      if (!this.vectorStore) {
        return { total: 0, documents: [] };
      }
      const store = this.vectorStore as MemoryVectorStore;
      const vectors = store.memoryVectors || [];
      const total = vectors.length;
      const offset = (page - 1) * pageSize;
      const paginated = vectors.slice(offset, offset + pageSize);
      
      const docs = paginated.map(v => new Document({
        pageContent: v.content,
        metadata: v.metadata,
      }));
      
      return { total, documents: docs };
    }

    // 2. Milvus 模式
    let client: any = null;
    try {
      // 动态导入 MilvusClient 并建立连接
      const { MilvusClient } = await import("@zilliz/milvus2-sdk-node");
      client = new MilvusClient({ address: this.milvusConfig.url! });

      // 健康检查
      const health = await client.checkHealth();
      if (!health.isHealthy) {
        throw new Error("Milvus is not healthy");
      }

      const collectionName = this.milvusConfig.collectionName!;
      
      // 集合检查
      const hasCollectionRes = await client.hasCollection({ collection_name: collectionName });
      if (!hasCollectionRes.value) {
        return { total: 0, documents: [] };
      }

      // 加载集合：查询前必须的操作
      await client.loadCollectionSync({ collection_name: collectionName });

      // 获取总数
      const statsRes = await client.getCollectionStatistics({ collection_name: collectionName });
      const rowCountItem = statsRes.stats.find((item: any) => item.key === "row_count");
      const total = rowCountItem ? parseInt(rowCountItem.value, 10) : 0;

      // 计算 offset
      const offset = (page - 1) * pageSize;
      const textField = this.milvusConfig.textField || "text";

      // 准备查询参数
      let queryParams = {
        collection_name: collectionName,
        filter: "langchain_primaryid >= 0", // 默认使用 langchain_primaryid
        output_fields: ["*"], // 返回所有标量字段
        limit: pageSize,
        offset: offset,
      };

      // 执行分页查询
      let res = await client.query(queryParams);

      // 主键兼容性处理
      if (res.status.error_code !== "Success") {
        console.warn(`[RAG] Query with 'langchain_primaryid' failed (${res.status.reason}), checking schema for actual primary key...`);
        
        // 调用 describeCollection 获取 Schema
        const describeRes = await client.describeCollection({ collection_name: collectionName });
        
        // 找到 is_primary_key: true 的字段名
        const pkFieldObj = describeRes.schema.fields.find((field: any) => field.is_primary_key === true);
        
        if (pkFieldObj && pkFieldObj.name !== "langchain_primaryid") {
          const actualPkField = pkFieldObj.name;
          console.log(`[RAG] Found actual primary key: '${actualPkField}', retrying query...`);
          
          // 使用新的主键名重试查询
          queryParams.filter = `${actualPkField} >= 0`;
          res = await client.query(queryParams);
        }
      }

      // 处理返回结果
      const docs = (res.data || []).map((item: any) => new Document({
        pageContent: item[textField],
        metadata: {
          source: item.source,
          ext: item.ext,
          // 保留所有其他字段，除了主键、向量和文本本身
          ...Object.keys(item).reduce((acc: any, key) => {
            if (key !== textField && key !== "vector" && !key.includes("id")) {
              acc[key] = item[key];
            }
            return acc;
          }, {})
        }
      }));

      return { total, documents: docs };
    } catch (error) {
      console.error("[RAG] Failed to fetch documents from Milvus:", error);
      // 根据要求，这里可以返回空列表或者抛出异常。
      // 为保持健壮性，捕获健康检查等异常后返回空列表。
      return { total: 0, documents: [] };
    } finally {
      // 资源清理：finally 中关闭连接
      if (client) {
        await client.closeConnection();
      }
    }
  }
  /**
   * 核心对话方法：完整的 检索-重排序-生成 (Retrieval-Rerank-Generation) 流程
   *
   * @param query 用户提问
   * @returns 包含生成的回复内容和引用的数据来源
   */
  async chat(query: string, history: {role: string, content: string}[] = []): Promise<{ answer: string; sources: Document[] }> {
    // 1. Mock 模式前置拦截
    // Mock Mode check - moved to top to bypass DB check
    if (DEEPSEEK_CONFIG.apiKey === 'mock-key') {
        console.log("[Chat] Mock Mode: 跳过检索、Rerank 和生成，返回模拟数据。");
        const mockDocs = [
            new Document({ pageContent: "Mock Doc 1", metadata: { source: "mock", score: 0.9, relevanceScore: 9.9 } }),
            new Document({ pageContent: "Mock Doc 2", metadata: { source: "mock", score: 0.8, relevanceScore: 8.8 } })
        ];
        return {
            answer: "这是模拟的回答：确实有降噪耳机（Mock Mode）。",
            sources: mockDocs
        };
    }

    // --- 阶段零：多轮对话查询重写 (Query Rewrite) ---
    let searchTargetQuery = query;
    let historyText = "";

    // 过滤掉刚刚发进来的最新一条（当前 query），只取前面的真正的历史
    const validHistory = history.filter((msg, idx) => {
        return !(idx === history.length - 1 && msg.role === 'user' && msg.content === query);
    });

    if (validHistory.length > 0) {
      console.log(`[RAG] Phase 0: Rewriting query based on ${validHistory.length} history turns...`);
      historyText = validHistory.map(m => `${m.role === 'user' ? '用户' : '客服'}: ${m.content}`).join('\n');
      
      const rewritePromptTemplate = PromptTemplate.fromTemplate(`
给定以下对话历史和一个后续问题，请将后续问题重写为一个独立的、信息完整的问题，使其可以在没有对话历史的情况下被理解。
如果后续问题已经很清晰，不需要上下文，请直接返回原问题。
不要回答问题，只返回重写后的句子，不要有任何多余的解释。

对话历史：
{history}

后续问题：{query}

重写后的独立问题：
`);
      try {
        const rewriteChain = rewritePromptTemplate.pipe(this.llm).pipe(new StringOutputParser());
        searchTargetQuery = await rewriteChain.invoke({ history: historyText, query: query });
        console.log(`[RAG] Query rewritten: "${query}" -> "${searchTargetQuery}"`);
      } catch (error) {
        console.warn("[RAG] Query rewrite failed, falling back to original query.", error);
      }
    }

    // 2. 知识库状态检查与重连
    if (!this.vectorStore) {
      if (this.isMemoryStore) {
        throw new Error("知识库为空。请先上传文件构建知识库。");
      }
      
      // 尝试重连 Milvus
      try {
        console.log("[RAG] Vector store not initialized, attempting to reconnect...");
        this.vectorStore = await Milvus.fromExistingCollection(
          this.embeddings,
          this.milvusConfig
        );
        this.isMemoryStore = false;
      } catch (error) {
        console.error("[RAG] Failed to connect to Milvus in chat method:", error);
        throw new Error("知识库连接失败。请确保已上传文件且 Milvus 服务正常运行。");
      }
    }

    // 3. 阶段一：初步检索 (Retrieval)
    console.log(`[RAG] Phase 1: Retrieving top 10 documents for query: "${searchTargetQuery}"`);
    const searchResults = await this.vectorStore.similaritySearchWithScore(searchTargetQuery, 10);
    const candidateDocs = searchResults.map(([doc]) => doc);

    if (candidateDocs.length === 0) {
      return {
        answer: "抱歉，在知识库中没有找到相关信息来回答您的问题。",
        sources: []
      };
    }

    // 4. 阶段二：LLM 重排序 (Rerank)
    console.log(`[RAG] Phase 2: Reranking ${candidateDocs.length} candidates using LLM...`);
    
    // 构造候选文档文本
    const candidatesText = candidateDocs.map((doc, idx) => `[文档ID: ${idx}]\n内容: ${doc.pageContent}`).join("\n\n");
    
    // 构造 Rerank Prompt
    const rerankPromptTemplate = PromptTemplate.fromTemplate(`
你是一个文档相关性评分专家。请根据用户的提问，评估以下候选文档与提问的相关性。
相关性评分范围为 0 到 10 分。0 分表示完全不相关，10 分表示非常相关且能直接回答提问。

用户提问: {query}

候选文档:
{candidates}

请务必返回一个纯 JSON 数组，包含每个文档的 ID 和评分，不要包含任何其他文字解释或 Markdown 代码块包裹。
输出格式要求必须严格如下：
[
  {{"id": 0, "score": 9.5}},
  {{"id": 1, "score": 2.1}}
]
`);
    
    let finalDocs: Document[] = [];
    
    try {
      const rerankPrompt = await rerankPromptTemplate.format({
        query: searchTargetQuery,
        candidates: candidatesText
      });
      
      const rerankResponse = await this.llm.invoke(rerankPrompt);
      const responseText = rerankResponse.content.toString();
      
      // 清理可能包含的 Markdown JSON 代码块
      const jsonStr = responseText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      
      const scores = JSON.parse(jsonStr) as { id: number, score: number }[];
      console.log(`[RAG] Rerank scores parsed:`, scores);
      
      // 将评分回填并过滤
      for (const item of scores) {
        if (item.id >= 0 && item.id < candidateDocs.length) {
          const doc = candidateDocs[item.id];
          doc.metadata = { ...doc.metadata, relevanceScore: item.score };
          
          // 只保留评分 >= 6 的文档
          if (item.score >= 6) {
            finalDocs.push(doc);
          }
        }
      }
      
      // 按分数降序排列，取 Top 3
      finalDocs.sort((a, b) => ((b.metadata.relevanceScore as number) || 0) - ((a.metadata.relevanceScore as number) || 0));
      finalDocs = finalDocs.slice(0, 3);
      
    } catch (error) {
      console.warn("[RAG] Rerank process failed or JSON parse error. Falling back to top 3 initial search results.", error);
      // 降级策略：使用原始检索结果的前 3 个，并标记 relevanceScore 为 0
      finalDocs = candidateDocs.slice(0, 3).map(doc => {
        doc.metadata = { ...doc.metadata, relevanceScore: 0 };
        return doc;
      });
    }

    // 兜底：如果 Rerank 后结果为空，强制使用原始 Top 1，避免无话可说
    if (finalDocs.length === 0 && candidateDocs.length > 0) {
      console.log("[RAG] No docs passed rerank threshold. Using top 1 as fallback.");
      const fallbackDoc = candidateDocs[0];
      fallbackDoc.metadata = { ...fallbackDoc.metadata, relevanceScore: -1 };
      finalDocs = [fallbackDoc];
    }

    // 5. 阶段三：最终生成 (Generation)
    console.log(`[RAG] Phase 3: Generating answer using ${finalDocs.length} top documents...`);
    
    // 上下文组装
    const contextText = finalDocs.map(doc => doc.pageContent).join("\n\n---\n\n");
    
    const generatePromptTemplate = PromptTemplate.fromTemplate(`
你是"睿智商城"的智能客服。请基于以下提供的已知信息，友好、专业地回答用户的问题。
要求：
1. 只能基于提供的已知信息回答，不要编造任何内容。
2. 如果已知信息无法回答用户的问题，请直接告知用户你不知道或建议联系人工客服。
3. 语气要亲切、专业。

已知信息:
{context}

历史对话:
{history}

用户当前提问: {question}

回答:
`);

    const chain = generatePromptTemplate.pipe(this.llm).pipe(new StringOutputParser());
    
    const answer = await chain.invoke({
      context: contextText,
      history: historyText || "无",
      question: query
    });

    console.log("[RAG] Generation complete.");

    // 6. 返回结果
    return {
      answer,
      sources: finalDocs
    };
  }

}

/**
 * 声明全局单例挂载点。
 *
 * 目的：
 * - 在 Next.js 开发环境热更新时复用同一个实例。
 * - 避免每次热重载重复初始化模型/连接。
 */
declare global {
  var __ragEnginePromise: Promise<RAGEngine> | undefined;
}

/**
 * 获取 RAGEngine 单例。
 *
 * 首次调用时创建实例并完成异步初始化，后续直接复用同一个 Promise。
 */
export async function getRAGEngine(): Promise<RAGEngine> {
  if (!globalThis.__ragEnginePromise) {
    console.log("[RAG Singleton] creating new promise", {
      pid: process.pid,
      node: process.version,
    });
    globalThis.__ragEnginePromise = (async () => {
      const engine = new RAGEngine();
      // 单例首次创建时立即完成初始化，避免调用方拿到未初始化实例。
      await engine.init();
      return engine;
    })();
  } else {
    console.log("[RAG Singleton] reusing existing promise", {
      pid: process.pid,
      node: process.version,
    });
  }
  return globalThis.__ragEnginePromise;
}
