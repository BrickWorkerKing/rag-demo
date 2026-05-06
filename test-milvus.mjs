import { MilvusClient } from "@zilliz/milvus2-sdk-node";
async function test() {
  console.log("Connecting...");
  const client = new MilvusClient({ address: "127.0.0.1:19530" });
  try {
    const res = await client.checkHealth();
    console.log("Health:", res);
  } catch (e) {
    console.error("Direct Error:", e);
  }
}
test();
