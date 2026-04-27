import path from "node:path";

import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";
import { ChatOpenAI } from "@langchain/openai";
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

  constructor() {
    // 初始化本地 embedding（中文 bge small）。
    this.embeddings = new LocalHuggingFaceEmbeddings({
      model: "Xenova/bge-small-zh-v1.5",
      cacheDir: LOCAL_MODEL_DIR,
    });

    // 初始化 ChatOpenAI，使用你指定的模型与 baseURL。
    this.llm = new ChatOpenAI({
      model: "doubao-seed-1-6-flash-250828",
      apiKey: process.env.DEEPSEEK_API_KEY,
      configuration: {
        baseURL: "https://sg.uiuiapi.com/v1",
      },
    });

    // Milvus 默认连接本机 19530，可通过环境变量覆盖。
    this.milvusConfig = {
      collectionName: process.env.MILVUS_COLLECTION_NAME ?? "rag_demo",
      url: process.env.MILVUS_URL ?? "http://127.0.0.1:19530",
      primaryField: "id",
      vectorField: "vector",
      textField: "text",
      autoId: true,
    };

    if (!process.env.DEEPSEEK_API_KEY) {
      console.warn(
        "[RAG] DEEPSEEK_API_KEY is not set. Chat model calls will fail until it is configured."
      );
    }
  }

  /**
   * 初始化向量库。
   *
   * 策略：
   * 1. 先尝试连接已有 Milvus 集合（fromExistingCollection）。
   * 2. 如果失败，自动降级到 MemoryVectorStore。
   *
   * 注意：
   * - 降级后 isMemoryStore=true，后续入库将走内存分支。
   */
  async init(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      try {
        console.log("[RAG] Trying to connect to existing Milvus collection...");
        this.vectorStore = await Milvus.fromExistingCollection(
          this.embeddings,
          this.milvusConfig
        );
        this.isMemoryStore = false;
        console.log("[RAG] Milvus connected.");
      } catch (error) {
        console.warn(
          "[RAG] Milvus connection failed. Fallback to in-memory vector store.",
          error
        );
        this.isMemoryStore = true;
        this.vectorStore = new MemoryVectorStore(this.embeddings);
      }
    })();

    return this.initPromise;
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
   * 文档入库主流程（ETL）。
   *
   * 完整步骤：
   * 1. init：确保向量库可用（Milvus 或内存降级）。
   * 2. load：按文件类型加载为 Document[]。
   * 3. split：使用 RecursiveCharacterTextSplitter 切块。
   * 4. embed/store：
   *    - 未创建向量库时：优先创建 Milvus，失败降级 Memory。
   *    - 已有向量库时：直接 addDocuments 追加。
   */
  async addDocument(fileName: string, fileBuffer: Buffer): Promise<void> {
    await this.init();

    const rawDocs = await this.loadDocuments(fileName, fileBuffer);
    if (rawDocs.length === 0) {
      console.warn("[RAG] No documents parsed, skip indexing.");
      return;
    }

    // chunkSize 按你的要求设置为 800；保留一定 overlap 提升召回连续性。
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 100,
    });

    // 给每个文档补充统一 source，方便后续检索结果溯源。
    const docs = rawDocs.map((doc) => {
      return new Document({
        pageContent: doc.pageContent,
        metadata: {
          ...doc.metadata,
          source: fileName,
        },
      });
    });

    const chunks = await splitter.splitDocuments(docs);
    console.log(
      `[RAG] Parsed ${rawDocs.length} docs from ${fileName}, split into ${chunks.length} chunks.`
    );

    // 首次入库：如果还没有 vectorStore，则先创建。
    if (!this.vectorStore) {
      // 优先走 Milvus 创建分支。
      if (!this.isMemoryStore) {
        try {
          console.log("[RAG] Creating Milvus collection via fromDocuments...");
          this.vectorStore = await Milvus.fromDocuments(
            chunks,
            this.embeddings,
            this.milvusConfig
          );
          console.log("[RAG] Milvus collection created and seeded.");
          return;
        } catch (error) {
          console.warn(
            "[RAG] Milvus create failed. Falling back to MemoryVectorStore.",
            error
          );
          this.isMemoryStore = true;
        }
      }

      // Milvus 不可用时，降级为内存向量库并完成首批数据写入。
      this.vectorStore = await MemoryVectorStore.fromDocuments(
        chunks,
        this.embeddings
      );
      console.log("[RAG] MemoryVectorStore created and seeded.");
      return;
    }

    // 非首次入库：直接追加文档。
    try {
      await this.vectorStore.addDocuments(chunks);
      console.log(`[RAG] Added ${chunks.length} chunks into vector store.`);
    } catch (error) {
      // 如果当前是 Milvus 且追加失败，自动切到内存模式，保证流程不中断。
      if (!this.isMemoryStore) {
        console.warn(
          "[RAG] Milvus addDocuments failed. Switching to MemoryVectorStore.",
          error
        );
        this.vectorStore = await MemoryVectorStore.fromDocuments(
          chunks,
          this.embeddings
        );
        this.isMemoryStore = true;
        return;
      }

      // 已经是内存模式仍失败，向上抛错让调用方处理。
      throw error;
    }
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
  var __ragEngineSingleton: RAGEngine | undefined;
}

/**
 * 获取 RAGEngine 单例。
 *
 * 首次调用创建实例，后续直接复用。
 */
export function getRAGEngine(): RAGEngine {
  if (!globalThis.__ragEngineSingleton) {
    globalThis.__ragEngineSingleton = new RAGEngine();
  }
  return globalThis.__ragEngineSingleton;
}

/** 便捷导出：可直接 import { ragEngine } 使用 */
export const ragEngine = getRAGEngine();
