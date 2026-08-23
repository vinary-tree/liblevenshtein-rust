package liblevenshtein

// C9 leak-discipline suite for the Go facade, measured with runtime.ReadMemStats.
// A >=10,000-cycle create/use/free loop must reach a heap steady state. Native
// handles are freed deterministically at Close(); this asserts the Go heap
// (HeapAlloc after a forced GC) does not drift upward across the cycles, which
// would reveal a retained iterator, transducer, or dictionary wrapper.

import (
	"runtime"
	"testing"

	lib "github.com/vinary-tree/libdictenstein/bindings/go/v4"
)

const (
	leakCycles = 10_000
	leakWarmup = 2_000
	// A per-cycle leak would accrue far more than this ceiling over 10k cycles.
	maxHeapGrowthBytes = 8 << 20
)

func settledHeapAlloc() uint64 {
	var stats runtime.MemStats
	for i := 0; i < 3; i++ {
		runtime.GC()
	}
	runtime.ReadMemStats(&stats)
	return stats.HeapAlloc
}

func assertSteady(t *testing.T, label string, cycle func()) {
	t.Helper()
	for i := 0; i < leakWarmup; i++ {
		cycle()
	}
	base := settledHeapAlloc()
	for i := 0; i < leakCycles; i++ {
		cycle()
	}
	growth := int64(settledHeapAlloc()) - int64(base)
	if growth > maxHeapGrowthBytes {
		t.Fatalf("%s: heap grew %d bytes over %d cycles", label, growth, leakCycles)
	}
}

func TestTransducerCycleHasNoLeak(t *testing.T) {
	values := []struct {
		term  string
		value *uint64
	}{
		{"cat", ptr(1)}, {"cot", ptr(2)}, {"cut", ptr(3)}, {"scat", nil},
	}
	assertSteady(t, "transducer iterator", func() {
		dawg, err := lib.NewDynamicDawg(lib.UnicodeScalarDomain)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range values {
			if _, err := dawg.Put(entry.term, entry.value); err != nil {
				t.Fatal(err)
			}
		}
		transducer, err := NewTransducer(dawg, Standard)
		if err != nil {
			t.Fatal(err)
		}
		iterator, err := transducer.Query("cat", 2, Traversal)
		if err != nil {
			t.Fatal(err)
		}
		for {
			_, ok, err := iterator.Next()
			if err != nil {
				t.Fatal(err)
			}
			if !ok {
				break
			}
		}
		_ = iterator.Close()
		_ = transducer.Close()
		_ = dawg.Close()
	})
}

func TestPhoneticCycleHasNoLeak(t *testing.T) {
	assertSteady(t, "phonetic pattern", func() {
		pattern, err := CompilePhoneticRegex("c[ao]t")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pattern.Matches("cat"); err != nil {
			t.Fatal(err)
		}
		_ = pattern.Close()
	})
}

func TestPhoneticRulesCycleHasNoLeak(t *testing.T) {
	assertSteady(t, "phonetic rules", func() {
		rules, err := BuiltinPhoneticRules(EnglishOrthography)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := rules.Apply("phone"); err != nil {
			t.Fatal(err)
		}
		_ = rules.Close()
	})
}

func ptr(value uint64) *uint64 { return &value }
