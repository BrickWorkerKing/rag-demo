import { NextRequest, NextResponse } from "next/server";
import { getRAGEngine } from "@/lib/rag";

export async function GET(req: NextRequest) {
  try {
    const searchParams = req.nextUrl.searchParams;
    const page = parseInt(searchParams.get("page") || "1", 10);
    const pageSize = parseInt(searchParams.get("pageSize") || "10", 10);

    if (isNaN(page) || page < 1) {
      return NextResponse.json({ error: "Invalid page parameter" }, { status: 400 });
    }
    if (isNaN(pageSize) || pageSize < 1) {
      return NextResponse.json({ error: "Invalid pageSize parameter" }, { status: 400 });
    }

    const rag = await getRAGEngine();
    const result = await rag.getDocuments(page, pageSize);

    return NextResponse.json({
      success: true,
      total: result.total,
      page,
      pageSize,
      documents: result.documents,
    });
  } catch (error: any) {
    console.error("[API] Fetch documents error:", error);
    return NextResponse.json(
      { error: error.message || "An unexpected error occurred while fetching documents" },
      { status: 500 }
    );
  }
}
