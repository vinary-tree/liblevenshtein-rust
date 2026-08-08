import { copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const runtimeRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const repositoryRoot = dirname(dirname(runtimeRoot));

copyFileSync(join(repositoryRoot, "LICENSE"), join(runtimeRoot, "LICENSE"));
