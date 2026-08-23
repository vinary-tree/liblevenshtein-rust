"use strict";

const Module = require("node:module");
const { resolve } = require("node:path");

process.env.NODE_PATH = resolve("node_modules");
Module._initPaths();
const compiledTest = resolve("../../target/liblevenshtein-cljs/main.cjs");
require(compiledTest);
