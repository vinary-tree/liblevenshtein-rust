import dictionary, { type Dictionary } from "@vinary-tree/libdictenstein/typescript";
import levenshtein, { type Match, type QueryCursor } from "@vinary-tree/liblevenshtein/typescript";

const words: Dictionary = dictionary.dynamicDawg("unicode");
words.put("cat", 1n);
const automaton = levenshtein.transducer(words, "damerau-levenshtein");
const cursor: QueryCursor = automaton.query("cat", 2, "distance-then-term");
const first: IteratorResult<Match> = cursor.next();
if (!first.done) first.value.term.domain satisfies "unicode" | "byte" | "u64";
cursor.reduceBatches((count, batch) => count + batch.length, 0);
cursor.close();
automaton.close();
words.close();
