#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: analyze-wallbreaker-oracle-control.R <raw-output-dir>")
}

files <- list.files(
  args[[1]],
  pattern = "^(selective|short)-(wallbreaker|exhaustive)-[0-9]+\\.txt$",
  full.names = TRUE
)
if (length(files) != 128L) stop("expected exactly 128 measured process outputs")

parse_file <- function(file) {
  lines <- readLines(file, warn = FALSE)
  if (length(lines) != 1L) stop("expected one record in ", file)
  fields <- strsplit(lines[[1]], " ", fixed = TRUE)[[1]]
  pairs <- strsplit(fields, "=", fixed = TRUE)
  if (any(lengths(pairs) != 2L)) stop("malformed record in ", file)
  record <- setNames(vapply(pairs, `[[`, "", 2L), vapply(pairs, `[[`, "", 1L))
  data.frame(
    case = record[["case"]],
    arm = record[["arm"]],
    seed = as.integer(record[["seed"]]),
    iterations = as.integer(record[["iterations"]]),
    result_count = as.integer(record[["result_count"]]),
    latency_ns = as.numeric(record[["ns_per_query"]]),
    allocations = as.numeric(record[["allocations_per_query"]]),
    allocated_bytes = as.numeric(record[["bytes_per_query"]])
  )
}

raw <- do.call(rbind, lapply(files, parse_file))
if (any(raw$iterations != 500L) || any(!is.finite(raw$latency_ns)) ||
    any(raw$latency_ns <= 0) || any(raw$allocations < 0) ||
    any(raw$allocated_bytes < 0)) {
  stop("unexpected iteration count or invalid measurement")
}

paired <- reshape(
  raw,
  idvar = c("case", "seed"),
  timevar = "arm",
  direction = "wide"
)
paired <- paired[order(paired$case, paired$seed), ]
for (case in c("selective", "short")) {
  rows <- paired[paired$case == case, ]
  if (!identical(rows$seed, 1001:1032) ||
      any(rows$result_count.wallbreaker != rows$result_count.exhaustive)) {
    stop("missing seed pair or unequal result count in ", case)
  }
}

set.seed(20261002)
for (case in c("selective", "short")) {
  rows <- paired[paired$case == case, ]
  ratio <- rows$latency_ns.wallbreaker / rows$latency_ns.exhaustive
  block <- (rows$seed - 1001L) %/% 2L + 1L
  bootstrap <- replicate(20000L, {
    sampled <- sample.int(16L, 16L, replace = TRUE)
    mean(unlist(lapply(sampled, function(id) ratio[block == id])))
  })
  ci90 <- unname(quantile(bootstrap, c(0.05, 0.95), names = FALSE))
  cat(sprintf(
    "%s n=32 mean_latency_ratio=%.9f ci90=[%.9f,%.9f] median_latency_ratio=%.9f min_ratio=%.9f max_ratio=%.9f mean_wallbreaker_ns=%.3f mean_exhaustive_ns=%.3f mean_wallbreaker_allocations=%.3f mean_exhaustive_allocations=%.3f mean_wallbreaker_bytes=%.3f mean_exhaustive_bytes=%.3f\n",
    case, mean(ratio), ci90[1], ci90[2], median(ratio), min(ratio),
    max(ratio), mean(rows$latency_ns.wallbreaker),
    mean(rows$latency_ns.exhaustive), mean(rows$allocations.wallbreaker),
    mean(rows$allocations.exhaustive), mean(rows$allocated_bytes.wallbreaker),
    mean(rows$allocated_bytes.exhaustive)
  ))
}
