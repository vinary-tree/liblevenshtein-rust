package liblevenshtein

// C8 native property-based tests for the Go facade, driven by the standard
// library's testing/quick. Every property is checked against an in-language
// brute-force Levenshtein oracle over a random dictionary built with the
// libdictenstein sibling's DynamicDawg (resolved through the repo go.work),
// which implements the same resource interface the Transducer consumes.
//
// Properties:
//   (a) distance symmetry d(a,b)==d(b,a), identity d(a,a)==0, and threshold
//       consistency dThreshold(a,b,k) == (d<=k ? d : sentinel) across the
//       exposed Levenshtein/Damerau/true-Damerau variants, plus standard
//       agreement with the oracle;
//   (b) a query's result set at distance k equals {t in dict : lev(query,t)<=k},
//       with exact distances and value round-trips;
//   (c) u64 value-channel round-trips, with 0 and 2^64-1 pinned explicitly.
//
// Determinism: every quick.Check runs from a fixed PRNG seed.
//
// KNOWN FACADE DEFECT (reported, not fixed here per the test-only remit): the
// standalone distance functions return the exceeded-bound sentinel for an empty
// operand, because bytesPointer returns a nil pointer for a zero-length slice
// and the native distance entry point rejects a null data pointer. The
// transducer query path handles the same nil+0 correctly, so empties are only
// excluded from the distance-function properties, not the query/value ones.

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	lib "github.com/vinary-tree/libdictenstein/bindings/go"
)

const seed = 20260809

var alphabet = []rune{'a', 'b', 'c', 'é'}

func genString(rng *rand.Rand, minLength, maxLength int) string {
	length := minLength + rng.Intn(maxLength-minLength+1)
	runes := make([]rune, length)
	for i := range runes {
		runes[i] = alphabet[rng.Intn(len(alphabet))]
	}
	return string(runes)
}

type term string

func (term) Generate(rng *rand.Rand, _ int) reflect.Value {
	return reflect.ValueOf(term(genString(rng, 0, 6)))
}

type nonEmptyTerm string

func (nonEmptyTerm) Generate(rng *rand.Rand, _ int) reflect.Value {
	return reflect.ValueOf(nonEmptyTerm(genString(rng, 1, 6)))
}

type threshold int

func (threshold) Generate(rng *rand.Rand, _ int) reflect.Value {
	return reflect.ValueOf(threshold(rng.Intn(4)))
}

type entry struct {
	Term  string
	Value *uint64
}

type dictionary []entry

func (dictionary) Generate(rng *rand.Rand, _ int) reflect.Value {
	count := rng.Intn(9)
	entries := make(dictionary, 0, count)
	for i := 0; i < count; i++ {
		var value *uint64
		if rng.Intn(4) != 0 {
			candidate := rng.Uint64()
			value = &candidate
		}
		entries = append(entries, entry{Term: genString(rng, 0, 6), Value: value})
	}
	return reflect.ValueOf(entries)
}

func config(count int) *quick.Config {
	return &quick.Config{MaxCount: count, Rand: rand.New(rand.NewSource(seed))}
}

func oracle(a, b string) int {
	if a == b {
		return 0
	}
	left, right := []rune(a), []rune(b)
	if len(left) == 0 {
		return len(right)
	}
	if len(right) == 0 {
		return len(left)
	}
	previous := make([]int, len(right)+1)
	for j := range previous {
		previous[j] = j
	}
	for i := 1; i <= len(left); i++ {
		current := make([]int, len(right)+1)
		current[0] = i
		for j := 1; j <= len(right); j++ {
			cost := 0
			if left[i-1] != right[j-1] {
				cost = 1
			}
			current[j] = min3(previous[j]+1, current[j-1]+1, previous[j-1]+cost)
		}
		previous = current
	}
	return previous[len(right)]
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

func equalValue(a, b *uint64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func runQuery(entries map[string]*uint64, query string, maximumDistance uint) (map[string]Match, error) {
	dawg, err := lib.NewDynamicDawg(lib.UnicodeScalarDomain)
	if err != nil {
		return nil, err
	}
	defer dawg.Close()
	for term, value := range entries {
		if _, err := dawg.Put(term, value); err != nil {
			return nil, err
		}
	}
	transducer, err := NewTransducer(dawg, Standard)
	if err != nil {
		return nil, err
	}
	defer transducer.Close()
	iterator, err := transducer.Query(query, maximumDistance, Traversal)
	if err != nil {
		return nil, err
	}
	defer iterator.Close()
	result := make(map[string]Match)
	for {
		match, ok, err := iterator.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		result[match.Text] = match
	}
	return result, nil
}

// --- (a) distance self-consistency across all exposed variants --------------

func TestDistanceProperties(t *testing.T) {
	sentinel := ^uint(0) - 1
	variants := []struct {
		name      string
		distance  func(string, string) uint
		threshold func(string, string, uint) uint
	}{
		{"levenshtein", Distance, DistanceThreshold},
		{"damerau", DamerauDistance, DamerauDistanceThreshold},
		{"trueDamerau", TrueDamerauDistance, TrueDamerauDistanceThreshold},
	}
	for _, variant := range variants {
		variant := variant
		symmetry := func(a, b nonEmptyTerm) bool {
			return variant.distance(string(a), string(b)) == variant.distance(string(b), string(a)) &&
				variant.distance(string(a), string(a)) == 0
		}
		if err := quick.Check(symmetry, config(500)); err != nil {
			t.Fatalf("%s symmetry/identity: %v", variant.name, err)
		}
		consistency := func(a, b nonEmptyTerm, k threshold) bool {
			full := variant.distance(string(a), string(b))
			bounded := variant.threshold(string(a), string(b), uint(k))
			if full <= uint(k) {
				return bounded == full
			}
			return bounded == sentinel
		}
		if err := quick.Check(consistency, config(500)); err != nil {
			t.Fatalf("%s threshold: %v", variant.name, err)
		}
	}
	agreement := func(a, b nonEmptyTerm) bool {
		return int(Distance(string(a), string(b))) == oracle(string(a), string(b))
	}
	if err := quick.Check(agreement, config(500)); err != nil {
		t.Fatalf("standard oracle agreement: %v", err)
	}
}

// --- (b) query result set equals the oracle, (c) value round-trips ----------

func TestQueryMatchesOracle(t *testing.T) {
	property := func(dict dictionary, query term, k threshold) bool {
		entries := make(map[string]*uint64)
		for _, e := range dict {
			entries[e.Term] = e.Value
		}
		got, err := runQuery(entries, string(query), uint(k))
		if err != nil {
			t.Errorf("query failed: %v", err)
			return false
		}
		expected := make(map[string]*uint64)
		for term, value := range entries {
			if oracle(string(query), term) <= int(k) {
				expected[term] = value
			}
		}
		if len(got) != len(expected) {
			return false
		}
		for term, match := range got {
			value, ok := expected[term]
			if !ok {
				return false
			}
			if int(match.Distance) != oracle(string(query), term) {
				return false
			}
			if !equalValue(match.ID, value) {
				return false
			}
		}
		return true
	}
	if err := quick.Check(property, config(150)); err != nil {
		t.Fatalf("query vs oracle: %v", err)
	}
}

func TestU64ValueRoundTrip(t *testing.T) {
	for _, value := range []uint64{0, 1, 1 << 63, ^uint64(0)} {
		v := value
		got, err := runQuery(map[string]*uint64{"term": &v}, "term", 0)
		if err != nil {
			t.Fatalf("value %d: %v", value, err)
		}
		match, ok := got["term"]
		if !ok || match.ID == nil || *match.ID != value {
			t.Fatalf("value %d did not round-trip: %+v", value, match)
		}
	}
	property := func() bool {
		candidate := rand.New(rand.NewSource(seed)).Uint64()
		got, err := runQuery(map[string]*uint64{"term": &candidate}, "term", 0)
		if err != nil {
			t.Errorf("query failed: %v", err)
			return false
		}
		match, ok := got["term"]
		return ok && match.ID != nil && *match.ID == candidate
	}
	if err := quick.Check(property, config(200)); err != nil {
		t.Fatalf("u64 round-trip: %v", err)
	}
}
