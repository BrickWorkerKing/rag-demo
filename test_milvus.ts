import { Document } from "@langchain/core/documents";
import { LocalHuggingFaceEmbeddings } from "./src/lib/rag";
import { Milvus } from "@langchain/community/vectorstores/milvus";

async function main() {
    const embeddings = new LocalHuggingFaceEmbeddings({
      model: "Xenova/bge-small-zh-v1.5",
      cacheDir: "./models",
    });
    const docs = [new Document({ pageContent: "test", metadata: { source: "test.csv" } })];
    
    console.log("Embedding documents...");
    const vectors = await embeddings.embedDocuments(["test"]);
    console.log("Vector dim:", vectors[0].length);

    console.log("Connecting to Milvus...");
    try {
        await Milvus.fromDocuments(docs, embeddings, {
            collectionName: "test_collection_123",
            url: "http://127.0.0.1:19530",
            primaryField: "id",
            vectorField: "vector",
            textField: "text",
            autoId: true,
        });
        console.log("Success");
    } catch (e) {
        console.error("Error:", e);
    }
}
main();
