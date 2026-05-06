import { MilvusClient } from "@zilliz/milvus2-sdk-node";

async function test() {
  try {
    const client = new MilvusClient({ address: "127.0.0.1:19530" });
    const health = await client.checkHealth();
    console.log("health:", health);
  } catch (e) {
    console.error("FAILED:", e);
  }
}

test();