"use strict";

const Module = require("node:module");
const { resolve } = require("node:path");

process.env.NODE_PATH = resolve("node_modules");
Module._initPaths();
process.chdir("/");
require("/tmp/vinary-tree-liblevenshtein-cljs/main.cjs");
