#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: analyze-wallbreaker-short-streaming.R <raw-output-dir>")
}

iterations_file <- file.path(args[[1]], "iterations.txt")
if (file.exists(iterations_file)) {
  iteration_lines <- readLines(iterations_file, warn = FALSE)
  if (length(iteration_lines) != 1L ||
      !grepl("^[1-9][0-9]*$", iteration_lines[[1]])) {
    stop("invalid timed-iteration metadata")
  }
  expected_iterations <- as.integer(iteration_lines[[1]])
  if (is.na(expected_iterations)) stop("timed-iteration count exceeds R integer range")
} else {
  # Archived experiment #388 predates the metadata file and used 500 exactly.
  expected_iterations <- 500L
}

files <- list.files(
  args[[1]],
  pattern = "^(short|selective|empty)-(control|treatment)-[0-9]+\\.txt$",
  full.names = TRUE
)
if (length(files) != 192L) stop("expected exactly 192 measured process outputs")

parse_file <- function(file) {
  name <- basename(file)
  parts <- regmatches(name, regexec(
    "^(short|selective|empty)-(control|treatment)-([0-9]+)\\.txt$", name
  ))[[1]]
  if (length(parts) != 4L) stop("malformed filename: ", name)
  lines <- readLines(file, warn = FALSE)
  if (length(lines) != 1L) stop("expected one record in ", file)
  pairs <- strsplit(strsplit(lines[[1]], " ", fixed = TRUE)[[1]], "=", fixed = TRUE)
  if (any(lengths(pairs) != 2L)) stop("malformed record in ", file)
  record <- setNames(vapply(pairs, `[[`, "", 2L), vapply(pairs, `[[`, "", 1L))
  expected_arm <- if (parts[[2]] == "empty" && parts[[3]] == "control") {
    "eager"
  } else {
    "wallbreaker"
  }
  if (!identical(record[["case"]], parts[[2]]) ||
      !identical(record[["arm"]], expected_arm) ||
      as.integer(record[["seed"]]) != as.integer(parts[[4]])) {
    stop("record identity does not match filename: ", file)
  }
  data.frame(
    case = parts[[2]],
    arm = parts[[3]],
    seed = as.integer(parts[[4]]),
    iterations = as.integer(record[["iterations"]]),
    result_count = as.integer(record[["result_count"]]),
    latency_ns = as.numeric(record[["ns_per_query"]]),
    allocations = as.numeric(record[["allocations_per_query"]]),
    allocated_bytes = as.numeric(record[["bytes_per_query"]])
  )
}

raw <- do.call(rbind, lapply(files, parse_file))
if (any(raw$iterations != expected_iterations) || any(!is.finite(raw$latency_ns)) ||
    any(raw$latency_ns <= 0) || any(raw$allocations < 0) ||
    any(raw$allocated_bytes < 0)) {
  stop("unexpected iteration count or invalid measurement")
}

cat(sprintf("timed_iterations_per_process=%d\n", expected_iterations))

paired <- reshape(raw, idvar = c("case", "seed"), timevar = "arm", direction = "wide")
paired <- paired[order(paired$case, paired$seed), ]
for (case in c("short", "selective", "empty")) {
  rows <- paired[paired$case == case, ]
  if (!identical(rows$seed, 1001:1032) || nrow(rows) != 32L ||
      any(rows$result_count.control != rows$result_count.treatment)) {
    stop("missing seed pair or unequal result count in ", case)
  }
}

set.seed(20261002)
results <- list()
for (case in c("short", "selective", "empty")) {
  rows <- paired[paired$case == case, ]
  ratio <- rows$latency_ns.treatment / rows$latency_ns.control
  block <- (rows$seed - 1001L) %/% 2L + 1L
  bootstrap <- replicate(20000L, {
    sampled <- sample.int(16L, 16L, replace = TRUE)
    mean(unlist(lapply(sampled, function(id) ratio[block == id])))
  })
  ci90 <- unname(quantile(bootstrap, c(0.05, 0.95), names = FALSE))
  bytes_ratio <- mean(rows$allocated_bytes.treatment) /
    mean(rows$allocated_bytes.control)
  results[[case]] <- list(mean_ratio = mean(ratio), ci90 = ci90,
                          bytes_ratio = bytes_ratio)
  cat(sprintf(
    "%s n=32 mean_latency_ratio=%.9f ci90=[%.9f,%.9f] median_latency_ratio=%.9f min_ratio=%.9f max_ratio=%.9f mean_treatment_ns=%.3f mean_control_ns=%.3f mean_treatment_allocations=%.3f mean_control_allocations=%.3f mean_treatment_bytes=%.3f mean_control_bytes=%.3f bytes_ratio=%.6f\n",
    case, mean(ratio), ci90[1], ci90[2], median(ratio), min(ratio),
    max(ratio), mean(rows$latency_ns.treatment),
    mean(rows$latency_ns.control), mean(rows$allocations.treatment),
    mean(rows$allocations.control), mean(rows$allocated_bytes.treatment),
    mean(rows$allocated_bytes.control), bytes_ratio
  ))
}

short_pass <- results$short$mean_ratio < 1 && results$short$ci90[[2]] < 1 &&
  results$short$bytes_ratio <= 0.25
selective_pass <- results$selective$ci90[[2]] <= 1.05
cat(sprintf("predeclared_short_gate=%s predeclared_selective_gate=%s\n",
            short_pass, selective_pass))
if (!short_pass || !selective_pass) quit(status = 1L)
