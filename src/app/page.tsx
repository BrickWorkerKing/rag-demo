export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-4xl flex-col gap-4 p-6">
      <h1 className="text-2xl font-semibold">RAG Customer Service Chat</h1>
      <section className="flex flex-1 flex-col rounded-lg border border-gray-200 p-4">
        <div className="mb-4 text-sm text-gray-500">TODO: Render conversation messages here.</div>
        <div className="mt-auto flex gap-2">
          <input
            className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm"
            placeholder="Type your message..."
          />
          <button className="rounded-md bg-black px-4 py-2 text-sm text-white">Send</button>
        </div>
      </section>
    </main>
  );
}
