import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({ message: 'TODO: rag demo endpoint' }, { status: 501 });
}
