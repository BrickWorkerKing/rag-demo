import { NextRequest, NextResponse } from "next/server";
import { getRAGEngine } from "@/lib/rag";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json(
        { error: "No file uploaded. Please provide a 'file' field." },
        { status: 400 }
      );
    }

    const fileName = file.name;
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const rag = await getRAGEngine();
    // 调用 RAGE 引擎添加文档到 Milvus 数据库
    const chunksProcessed = await rag.addDocument(fileName, buffer);

    return NextResponse.json({
      success: true,
      message: "File uploaded and processed successfully",
      fileName: fileName,
      processedChunks: chunksProcessed,
    });
  } catch (error: any) {
    console.error("[API] Upload error:", error);
    return NextResponse.json(
      { error: error.message || "An unexpected error occurred during upload" },
      { status: 500 }
    );
  }
}
