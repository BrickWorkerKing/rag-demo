import { NextRequest, NextResponse } from "next/server";
import { getRAGEngine } from "@/lib/rag";

export async function POST(req: NextRequest) {
  try {
    const rag = await getRAGEngine();
    await rag.reset();

    return NextResponse.json({
      success: true,
      message: "Knowledge base has been successfully reset.",
    });
  } catch (error: any) {
    console.error("[API] Reset error:", error);
    return NextResponse.json(
      { error: error.message || "An unexpected error occurred while resetting knowledge base" },
      { status: 500 }
    );
  }
}
