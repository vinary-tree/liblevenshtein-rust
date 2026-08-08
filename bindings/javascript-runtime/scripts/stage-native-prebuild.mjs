import { copyFileSync, mkdirSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const platform = `${process.platform}-${process.arch}`;
const source = join(root, "native", "build", "Release", "vinary_tree_native.node");
const destination = join(
  root,
  "native",
  "prebuilds",
  platform,
  "vinary_tree_native.node",
);

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);

// The Rust release profile intentionally retains debug information for local
// profiling.  Published Node addons must not carry those symbols: they make a
// single platform artifact hundreds of megabytes larger without changing the
// executable code.  Windows keeps debug information in a separate PDB, so its
// PE addon needs no corresponding post-processing here.
if (process.platform === "linux") {
  execFileSync("strip", ["--strip-unneeded", destination], { stdio: "inherit" });
} else if (process.platform === "darwin") {
  execFileSync("strip", ["-S", destination], { stdio: "inherit" });
}
console.log(`staged ${platform} native addon at ${destination}`);
