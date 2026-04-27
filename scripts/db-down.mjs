import { execSync } from "node:child_process";

if (process.platform !== "win32") {
  console.error("db:down currently supports Windows only.");
  process.exit(1);
}

let found = false;

for (const image of ["milvus.exe", "milvus-server.exe"]) {
  try {
    execSync(`taskkill /F /IM ${image} /T`, { stdio: "ignore" });
    found = true;
  } catch {
    // Ignore missing process errors.
  }
}

if (found) {
  console.log("Milvus stop command sent.");
} else {
  console.log("No Milvus process found.");
}
