import { pipeline, env } from "@xenova/transformers"; 
import path from "path"; 
// 只配置 host，不配置 template 
env.remoteHost = "https://hf-mirror.com"; 
env.allowRemoteModels = true; 
async function download() { 
  const cacheDir = path.resolve("./models"); 
  console.log(`Downloading to: ${cacheDir}`); 
  await pipeline("feature-extraction", "Xenova/bge-small-zh-v1.5", { 
    cache_dir: cacheDir 
  }); 
  console.log("Download complete!"); 
} 
download().catch(console.error);