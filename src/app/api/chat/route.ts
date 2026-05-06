import { NextRequest, NextResponse } from "next/server";
import { getRAGEngine } from "@/lib/rag";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { query, history } = body;

    // 参数校验：query 为空则返回 400
    if (!query || typeof query !== "string" || query.trim() === "") {
      return NextResponse.json(
        { error: "参数 query 不能为空" },
        { status: 400 }
      );
    }

    // 调用引擎
    const rag = await getRAGEngine();
    const result = await rag.chat(query.trim(), history || []);

    // 返回结果
    return NextResponse.json({
      answer: result.answer,
      sources: result.sources,
    });
  } catch (error: any) {
    console.error("[API] Chat error:", error);
    return NextResponse.json(
      { error: error.message || "服务器内部错误，请稍后再试" },
      { status: 500 }
    );
  }
}
