import { NextResponse } from 'next/server';

export async function POST() {
  return NextResponse.json({ message: 'TODO: chat endpoint' }, { status: 501 });
}
