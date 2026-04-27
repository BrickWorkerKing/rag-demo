import net from "node:net";

const host = "127.0.0.1";
const port = 19530;

const socket = net.createConnection({ host, port });
socket.setTimeout(1500);

socket.on("connect", () => {
  console.log(`Milvus is running on ${host}:${port}`);
  socket.end();
  process.exit(0);
});

const reportDown = () => {
  console.error(`Milvus is not running on ${host}:${port}`);
  process.exit(1);
};

socket.on("timeout", () => {
  socket.destroy();
  reportDown();
});

socket.on("error", reportDown);
