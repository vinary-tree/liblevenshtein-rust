package io.vinarytree.liblevenshtein;

import java.util.OptionalLong;

/** One safely materialized fuzzy match. */
public record Match(Term term, long distance, OptionalLong id) {}
