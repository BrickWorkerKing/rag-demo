import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  allowedDevOrigins: ["172.16.1.199"],
  serverExternalPackages: [
    "@zilliz/milvus2-sdk-node",
    "@grpc/grpc-js",
    "@grpc/proto-loader",
    "protobufjs"
  ],
};

export default nextConfig;
