package liblevenshtein

import (
	"bytes"
	"errors"
	"reflect"
	"sort"
	"testing"

	lib "github.com/vinary-tree/libdictenstein/bindings/go/v4"
)

func mustTextDictionary(t *testing.T, entries map[string]*uint64) *lib.DynamicDawg {
	t.Helper()
	dictionary, err := lib.NewDynamicDawg(lib.UnicodeScalarDomain)
	if err != nil {
		t.Fatal(err)
	}
	for term, value := range entries {
		if _, err := dictionary.Put(term, value); err != nil {
			dictionary.Close()
			t.Fatal(err)
		}
	}
	return dictionary
}

func mustTransducer(t *testing.T, dictionary *lib.DynamicDawg, algorithm Algorithm) *Transducer {
	t.Helper()
	transducer, err := NewTransducer(dictionary, algorithm)
	if err != nil {
		t.Fatal(err)
	}
	return transducer
}

func drain(t *testing.T, iterator *Iterator) []Match {
	t.Helper()
	var matches []Match
	for {
		match, ok, err := iterator.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		matches = append(matches, match)
	}
	if err := iterator.Close(); err != nil {
		t.Fatal(err)
	}
	return matches
}

func containsText(matches []Match, term string, distance uint) bool {
	for _, match := range matches {
		if match.Text == term && match.Distance == distance {
			return true
		}
	}
	return false
}

func textTerms(matches []Match) []string {
	terms := make([]string, len(matches))
	for index, match := range matches {
		terms[index] = match.Text
	}
	return terms
}

func TestDistances(t *testing.T) {
	if got := Distance("kitten", "sitting"); got != 3 {
		t.Fatalf("distance=%d", got)
	}
	if got := DamerauDistance("ab", "ba"); got != 1 {
		t.Fatalf("Damerau=%d", got)
	}
	if got := TrueDamerauDistance("ca", "abc"); got != 2 {
		t.Fatalf("true Damerau=%d", got)
	}
}

func TestPhoneticPattern(t *testing.T) {
	pattern, err := CompilePhoneticRegex("cat")
	if err != nil {
		t.Fatal(err)
	}
	defer pattern.Close()
	accepted, err := pattern.Matches("cat")
	if err != nil || !accepted {
		t.Fatalf("accepted=%v err=%v", accepted, err)
	}
	rejected, err := pattern.Matches("cot")
	if err != nil || rejected {
		t.Fatalf("rejected=%v err=%v", rejected, err)
	}
}

// C6: the ABI counts Unicode scalar values, not bytes or UTF-16 code units, so
// the facade must pass the UTF-8 byte length while distances stay char-based.
func TestUnicodeDistances(t *testing.T) {
	// "café" is 5 bytes / 4 scalars; it differs from "cafe" by one scalar.
	if got := Distance("café", "cafe"); got != 1 {
		t.Fatalf("café/cafe distance=%d, want 1", got)
	}
	// A 4-byte astral codepoint is one unit (guards against a UTF-16 length bug).
	if got := Distance("🦀", "x"); got != 1 {
		t.Fatalf("🦀/x distance=%d, want 1", got)
	}
	// A combining mark is its own scalar: "e"+U+0301 vs "e" differs by one.
	if got := Distance("é", "e"); got != 1 {
		t.Fatalf("combining distance=%d, want 1", got)
	}
}

// C6: threshold functions pass the native sentinel (usize::MAX - 1) through
// unchanged when the exact distance exceeds the bound.
func TestThresholdSentinel(t *testing.T) {
	exceeded := ^uint(0) - 1 // native usize::MAX - 1
	if got := DistanceThreshold("kitten", "sitting", 3); got != 3 {
		t.Fatalf("within-bound=%d, want 3", got)
	}
	if got := DistanceThreshold("kitten", "sitting", 2); got != exceeded {
		t.Fatalf("exceeded=%d, want sentinel %d", got, exceeded)
	}
	// "ca" -> "abc": OSA is 3 but unrestricted Damerau-Levenshtein is 2.
	if got := DamerauDistanceThreshold("ca", "abc", 2); got != exceeded {
		t.Fatalf("OSA within 2 should exceed, got %d", got)
	}
	if got := TrueDamerauDistanceThreshold("ca", "abc", 2); got != 2 {
		t.Fatalf("true Damerau within 2=%d, want 2", got)
	}
}

// C2/C3: close is idempotent and using a closed handle is a returned error,
// never a crash or double-free.
func TestIdempotentClose(t *testing.T) {
	pattern, err := CompilePhoneticRegex("cat")
	if err != nil {
		t.Fatal(err)
	}
	if err := pattern.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := pattern.Close(); err != nil {
		t.Fatalf("second close must be a no-op, got %v", err)
	}
	if _, err := pattern.Matches("cat"); err == nil {
		t.Fatal("Matches on a closed pattern must return an error")
	}
	if _, err := pattern.Size(); err == nil {
		t.Fatal("Size on a closed pattern must return an error")
	}
}

func TestGeneratedEnumsAndTypedNativeError(t *testing.T) {
	if StatusDomainMismatch != 12 || DamerauLevenshtein != 3 ||
		DistanceThenTerm != 1 || EnglishPhonetic != 1 {
		t.Fatal("generated enum values drifted from the canonical ABI model")
	}
	unknown := Status(^uint32(0))
	if uint32(unknown) != ^uint32(0) {
		t.Fatal("forward-compatible unknown status was not preserved")
	}

	_, err := CompilePhoneticRegex("(")
	if err == nil {
		t.Fatal("invalid regex unexpectedly compiled")
	}
	var nativeError *Error
	if !errors.As(err, &nativeError) {
		t.Fatalf("error type=%T, want *Error", err)
	}
	if nativeError.Status != StatusInvalidArgument {
		t.Fatalf("status=%v, want StatusInvalidArgument", nativeError.Status)
	}
	if nativeError.Message == "" {
		t.Fatal("native diagnostic was not copied")
	}
}

func TestAlgorithmsOrdersAndIterator(t *testing.T) {
	values := map[string]*uint64{}
	for index, term := range []string{"ab", "c", "abc", "bat", "cat", "cats"} {
		value := uint64(index + 1)
		values[term] = &value
	}
	dictionary := mustTextDictionary(t, values)
	defer dictionary.Close()

	cases := []struct {
		algorithm Algorithm
		query     string
		distance  uint
		term      string
		expected  bool
	}{
		{Standard, "ba", 1, "ab", false},
		{Transposition, "ba", 1, "ab", true},
		{MergeAndSplit, "ab", 1, "c", true},
		{DamerauLevenshtein, "ca", 2, "abc", true},
	}
	for _, test := range cases {
		transducer := mustTransducer(t, dictionary, test.algorithm)
		iterator, err := transducer.Query(test.query, test.distance, Traversal)
		if err != nil {
			t.Fatal(err)
		}
		found := containsText(drain(t, iterator), test.term, test.distance)
		if found != test.expected {
			t.Fatalf("algorithm=%v term=%q found=%v", test.algorithm, test.term, found)
		}
		if err := transducer.Close(); err != nil {
			t.Fatal(err)
		}
	}

	transducer := mustTransducer(t, dictionary, Standard)
	defer transducer.Close()
	var order QueryOrder = DistanceThenTerm
	iterator, err := transducer.Query("cat", 1, order)
	if err != nil {
		t.Fatal(err)
	}
	if got := textTerms(drain(t, iterator)); !reflect.DeepEqual(got, []string{"cat", "bat", "cats"}) {
		t.Fatalf("distance order=%v", got)
	}
}

func TestByteAndU64QueriesPreserveDomainsAndValues(t *testing.T) {
	byteDictionary, err := lib.NewDynamicDawg(lib.ByteDomain)
	if err != nil {
		t.Fatal(err)
	}
	defer byteDictionary.Close()
	byteTerm := []byte{0xff, 0x00, 0x7f}
	byteValue := ^uint64(0)
	if _, err := byteDictionary.Put(string(byteTerm), &byteValue); err != nil {
		t.Fatal(err)
	}
	byteTransducer := mustTransducer(t, byteDictionary, Standard)
	defer byteTransducer.Close()
	byteIterator, err := byteTransducer.QueryBytes([]byte{0xff, 0x00, 0x7e}, 1)
	if err != nil {
		t.Fatal(err)
	}
	byteMatches := drain(t, byteIterator)
	if len(byteMatches) != 1 || byteMatches[0].Domain != ByteDomain ||
		!bytes.Equal(byteMatches[0].Bytes, byteTerm) || byteMatches[0].Distance != 1 ||
		byteMatches[0].ID == nil || *byteMatches[0].ID != byteValue {
		t.Fatalf("byte match=%+v", byteMatches)
	}

	tokenDictionary, err := lib.NewDynamicDawg(lib.U64Domain)
	if err != nil {
		t.Fatal(err)
	}
	defer tokenDictionary.Close()
	tokenTerm := []uint64{0, ^uint64(0)}
	tokenValue := uint64(7)
	if _, err := tokenDictionary.PutU64(tokenTerm, &tokenValue); err != nil {
		t.Fatal(err)
	}
	tokenTransducer := mustTransducer(t, tokenDictionary, Standard)
	defer tokenTransducer.Close()
	tokenIterator, err := tokenTransducer.QueryU64([]uint64{0, ^uint64(0) - 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	tokenMatches := drain(t, tokenIterator)
	if len(tokenMatches) != 1 || tokenMatches[0].Domain != U64Domain ||
		!reflect.DeepEqual(tokenMatches[0].Tokens, tokenTerm) ||
		tokenMatches[0].Distance != 1 || tokenMatches[0].ID == nil ||
		*tokenMatches[0].ID != tokenValue {
		t.Fatalf("u64 match=%+v", tokenMatches)
	}
}

func TestBoundedQueryCacheReportsHitsAndPreservesExactResults(t *testing.T) {
	dictionary := mustTextDictionary(t, map[string]*uint64{"cat": nil, "cot": nil})
	defer dictionary.Close()
	transducer := mustTransducer(t, dictionary, Standard)
	defer transducer.Close()
	cache, err := NewQueryCache(transducer, 8, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Close()
	coldCursor, err := cache.Query("cut", 1, Traversal)
	if err != nil {
		t.Fatal(err)
	}
	cold := drain(t, coldCursor)
	hitCursor, err := cache.Query("cut", 1, Traversal)
	if err != nil {
		t.Fatal(err)
	}
	hit := drain(t, hitCursor)
	if !reflect.DeepEqual(hit, cold) {
		t.Fatalf("cache changed exact result: cold=%v hit=%v", cold, hit)
	}
	stats, err := cache.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.Requests != 2 || stats.Hits != 1 || stats.Misses != 1 ||
		stats.ResidentEntries != 1 || stats.ResidentWeight == 0 {
		t.Fatalf("unexpected cache stats: %+v", stats)
	}
	if err := cache.ResetStats(); err != nil {
		t.Fatal(err)
	}
	stats, _ = cache.Stats()
	if stats.Requests != 0 || stats.ResidentEntries != 1 {
		t.Fatalf("reset changed residency: %+v", stats)
	}
	if err := cache.Clear(); err != nil {
		t.Fatal(err)
	}
	stats, _ = cache.Stats()
	if stats.ResidentEntries != 0 {
		t.Fatalf("clear retained results: %+v", stats)
	}
}

func TestQueryStartSnapshotSurvivesMutationAndProducerClose(t *testing.T) {
	values := map[string]*uint64{}
	for index, term := range []string{"cat", "cot", "cut", "scat"} {
		value := uint64(index + 1)
		values[term] = &value
	}
	dictionary := mustTextDictionary(t, values)
	transducer := mustTransducer(t, dictionary, Standard)
	defer transducer.Close()

	baselineCursor, err := transducer.Query("cat", 2, Traversal)
	if err != nil {
		t.Fatal(err)
	}
	baseline := textTerms(drain(t, baselineCursor))
	sort.Strings(baseline)

	iterator, err := transducer.Query("cat", 2, Traversal)
	if err != nil {
		t.Fatal(err)
	}
	first, ok, err := iterator.Next()
	if err != nil || !ok {
		t.Fatalf("first=%+v ok=%v err=%v", first, ok, err)
	}
	if err := dictionary.Clear(); err != nil {
		t.Fatal(err)
	}
	newValue := uint64(99)
	if _, err := dictionary.Put("new", &newValue); err != nil {
		t.Fatal(err)
	}
	if err := dictionary.Close(); err != nil {
		t.Fatal(err)
	}
	observed := append([]string{first.Text}, textTerms(drain(t, iterator))...)
	sort.Strings(observed)
	if !reflect.DeepEqual(observed, baseline) {
		t.Fatalf("snapshot=%v baseline=%v", observed, baseline)
	}

	fresh, err := transducer.Query("new", 0, Traversal)
	if err != nil {
		t.Fatal(err)
	}
	freshMatches := drain(t, fresh)
	if len(freshMatches) != 1 || freshMatches[0].Text != "new" {
		t.Fatalf("fresh snapshot=%+v", freshMatches)
	}
}

func TestPhoneticLLREProductAndRuleSets(t *testing.T) {
	llre, err := CompilePhoneticLLRE("@name \"Greeting\"\n^hello$")
	if err != nil {
		t.Fatal(err)
	}
	accepted, err := llre.Matches("hello")
	if err != nil || !accepted {
		t.Fatalf("LLRE accepted=%v err=%v", accepted, err)
	}
	if err := llre.Close(); err != nil {
		t.Fatal(err)
	}

	dictionary := mustTextDictionary(t, map[string]*uint64{
		"cat": nil, "cot": nil, "cut": nil,
	})
	defer dictionary.Close()
	transducer := mustTransducer(t, dictionary, Standard)
	defer transducer.Close()
	pattern, err := CompilePhoneticRegex("c[ao]t")
	if err != nil {
		t.Fatal(err)
	}
	defer pattern.Close()
	product, err := transducer.QueryPattern(pattern, 0)
	if err != nil {
		t.Fatal(err)
	}
	terms := textTerms(drain(t, product))
	sort.Strings(terms)
	if !reflect.DeepEqual(terms, []string{"cat", "cot"}) {
		t.Fatalf("product terms=%v", terms)
	}

	rules, err := ParsePhoneticRules("ph -> f\ngh ->\n")
	if err != nil {
		t.Fatal(err)
	}
	count, err := rules.Len()
	if err != nil || count != 2 {
		t.Fatalf("rule count=%d err=%v", count, err)
	}
	output, err := rules.Apply("phgh")
	if err != nil || output != "f" {
		t.Fatalf("rules output=%q err=%v", output, err)
	}
	if err := rules.Close(); err != nil {
		t.Fatal(err)
	}
	if err := rules.Close(); err != nil {
		t.Fatalf("idempotent rule close: %v", err)
	}

	var kind PhoneticRuleSetKind
	for _, kind = range []PhoneticRuleSetKind{EnglishOrthography, EnglishPhonetic} {
		builtin, err := BuiltinPhoneticRules(kind)
		if err != nil {
			t.Fatal(err)
		}
		count, err := builtin.Len()
		if err != nil || count == 0 {
			t.Fatalf("kind=%v count=%d err=%v", kind, count, err)
		}
		output, err := builtin.Apply("phone")
		if err != nil || output == "" {
			t.Fatalf("kind=%v output=%q err=%v", kind, output, err)
		}
		if err := builtin.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
