import { Milvus } from "@langchain/community/vectorstores/milvus";

async function test() {
  console.log("Testing Langchain Milvus...");
  
  try {
    const vectorStore = await Milvus.fromExistingCollection(
      { embedDocuments: async () => [[]], embedQuery: async () => [] },
      {
        collectionName: "rag_demo",
        url: "127.0.0.1:19530",
      }
    );
    console.log("Success:", vectorStore);
  } catch (e) {
    console.error("Langchain Milvus Error:", e);
  }
}
test();
