package liblevenshtein

import "testing"

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
