import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
const store = new MemoryVectorStore({} as any);
store.addDocuments([new Document({ pageContent: "hello", metadata: { a: 1 } })]).then(() => {
    console.log(store.memoryVectors);
});
