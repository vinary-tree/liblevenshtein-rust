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
