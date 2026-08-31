// Package liblevenshtein provides snapshot-consistent streaming fuzzy search.
package liblevenshtein

/*
#cgo CFLAGS: -std=c17 -I${SRCDIR}/../../include -I${SRCDIR}/../../../vinary-tree-interop/include
#cgo LDFLAGS: -lliblevenshtein
#include <stdint.h>
#include <stdlib.h>
#include "liblevenshtein.h"

static LlevStatus go_llev_transducer_new(uintptr_t context, uintptr_t vtable,
                                         uint32_t algorithm,
                                         LlevTransducer** output) {
    VtResource resource = {(void*)context, (const VtResourceVTable*)vtable};
    return llev_transducer_new(&resource, algorithm, output);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unicode/utf8"
	"unsafe"

	interop "github.com/vinary-tree/vinary-tree-interop/bindings/go/v4"
)

// UnitDomain identifies the native term representation.
type UnitDomain uint32

const (
	ByteDomain UnitDomain = iota + 1
	UnicodeScalarDomain
	U64Domain
)

// Error is a stable native ABI failure.
type Error struct {
	Status  Status
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("liblevenshtein status %d: %s", e.Status, e.Message)
}

func check(status C.LlevStatus) error {
	if status == C.LLEV_STATUS_OK {
		return nil
	}
	return &Error{Status: Status(status), Message: C.GoString(C.llev_last_error_message())}
}

func bytesPointer(value []byte) *C.uint8_t {
	if len(value) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&value[0]))
}

// Distance returns standard Unicode Levenshtein distance.
func Distance(source, target string) uint {
	left, right := []byte(source), []byte(target)
	result := C.llev_distance((*C.char)(unsafe.Pointer(bytesPointer(left))), C.size_t(len(left)), (*C.char)(unsafe.Pointer(bytesPointer(right))), C.size_t(len(right)))
	runtime.KeepAlive(left)
	runtime.KeepAlive(right)
	return uint(result)
}

// DistanceThreshold returns ^uint(0)-1 when the distance exceeds threshold.
func DistanceThreshold(source, target string, threshold uint) uint {
	left, right := []byte(source), []byte(target)
	result := C.llev_distance_threshold((*C.char)(unsafe.Pointer(bytesPointer(left))), C.size_t(len(left)), (*C.char)(unsafe.Pointer(bytesPointer(right))), C.size_t(len(right)), C.size_t(threshold))
	runtime.KeepAlive(left)
	runtime.KeepAlive(right)
	return uint(result)
}

// DamerauDistance recognizes adjacent transpositions.
func DamerauDistance(source, target string) uint {
	left, right := []byte(source), []byte(target)
	result := C.llev_damerau_distance((*C.char)(unsafe.Pointer(bytesPointer(left))), C.size_t(len(left)), (*C.char)(unsafe.Pointer(bytesPointer(right))), C.size_t(len(right)))
	runtime.KeepAlive(left)
	runtime.KeepAlive(right)
	return uint(result)
}

// DamerauDistanceThreshold returns ^uint(0)-1 when the adjacent-transposition
// distance exceeds threshold.
func DamerauDistanceThreshold(source, target string, threshold uint) uint {
	left, right := []byte(source), []byte(target)
	result := C.llev_damerau_distance_threshold((*C.char)(unsafe.Pointer(bytesPointer(left))), C.size_t(len(left)), (*C.char)(unsafe.Pointer(bytesPointer(right))), C.size_t(len(right)), C.size_t(threshold))
	runtime.KeepAlive(left)
	runtime.KeepAlive(right)
	return uint(result)
}

// TrueDamerauDistance computes unrestricted Damerau-Levenshtein distance.
func TrueDamerauDistance(source, target string) uint {
	left, right := []byte(source), []byte(target)
	result := C.llev_true_damerau_distance((*C.char)(unsafe.Pointer(bytesPointer(left))), C.size_t(len(left)), (*C.char)(unsafe.Pointer(bytesPointer(right))), C.size_t(len(right)))
	runtime.KeepAlive(left)
	runtime.KeepAlive(right)
	return uint(result)
}

// TrueDamerauDistanceThreshold returns ^uint(0)-1 when unrestricted distance
// exceeds threshold.
func TrueDamerauDistanceThreshold(source, target string, threshold uint) uint {
	left, right := []byte(source), []byte(target)
	result := C.llev_true_damerau_distance_threshold((*C.char)(unsafe.Pointer(bytesPointer(left))), C.size_t(len(left)), (*C.char)(unsafe.Pointer(bytesPointer(right))), C.size_t(len(right)), C.size_t(threshold))
	runtime.KeepAlive(left)
	runtime.KeepAlive(right)
	return uint(result)
}

// Transducer retains a modular dictionary resource in O(1).
type Transducer struct {
	mu      sync.RWMutex
	pointer *C.LlevTransducer
}

// NewTransducer constructs an automaton configuration over a dictionary.
func NewTransducer(dictionary interop.DictionaryResource, algorithm Algorithm) (*Transducer, error) {
	if dictionary == nil {
		return nil, errors.New("dictionary is nil")
	}
	var pointer *C.LlevTransducer
	err := dictionary.WithResource(func(context, vtable uintptr) error {
		return check(C.go_llev_transducer_new(C.uintptr_t(context), C.uintptr_t(vtable), C.uint32_t(algorithm), &pointer))
	})
	if err != nil {
		return nil, err
	}
	result := &Transducer{pointer: pointer}
	runtime.SetFinalizer(result, (*Transducer).finalize)
	return result, nil
}

func (t *Transducer) finalize() { _ = t.Close() }

// Close releases the transducer. Existing iterators retain their snapshots.
func (t *Transducer) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.pointer != nil {
		C.llev_transducer_free(t.pointer)
		t.pointer = nil
		runtime.SetFinalizer(t, nil)
	}
	return nil
}

func (t *Transducer) withPointer(call func(*C.LlevTransducer) (*C.LlevQueryCursor, error)) (*Iterator, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.pointer == nil {
		return nil, errors.New("transducer is closed")
	}
	cursor, err := call(t.pointer)
	if err != nil {
		return nil, err
	}
	return newIterator(cursor), nil
}

func newIterator(cursor *C.LlevQueryCursor) *Iterator {
	result := &Iterator{pointer: cursor}
	runtime.SetFinalizer(result, (*Iterator).finalize)
	return result
}

// Query starts a lazy Unicode query and captures the current dictionary revision.
func (t *Transducer) Query(query string, maximumDistance uint, order QueryOrder) (*Iterator, error) {
	if !utf8.ValidString(query) {
		return nil, errors.New("query is not valid UTF-8")
	}
	input := []byte(query)
	return t.withPointer(func(pointer *C.LlevTransducer) (*C.LlevQueryCursor, error) {
		var cursor *C.LlevQueryCursor
		status := C.llev_transducer_query_utf8(pointer, (*C.char)(unsafe.Pointer(bytesPointer(input))), C.size_t(len(input)), C.size_t(maximumDistance), C.uint32_t(order), &cursor)
		runtime.KeepAlive(input)
		return cursor, check(status)
	})
}

// QueryBytes starts a lazy raw-byte query.
func (t *Transducer) QueryBytes(query []byte, maximumDistance uint) (*Iterator, error) {
	return t.withPointer(func(pointer *C.LlevTransducer) (*C.LlevQueryCursor, error) {
		var cursor *C.LlevQueryCursor
		status := C.llev_transducer_query_bytes(pointer, bytesPointer(query), C.size_t(len(query)), C.size_t(maximumDistance), C.LLEV_QUERY_ORDER_TRAVERSAL, &cursor)
		runtime.KeepAlive(query)
		return cursor, check(status)
	})
}

// QueryU64 starts a lazy unsigned-token query.
func (t *Transducer) QueryU64(query []uint64, maximumDistance uint) (*Iterator, error) {
	return t.withPointer(func(pointer *C.LlevTransducer) (*C.LlevQueryCursor, error) {
		var data *C.uint64_t
		if len(query) != 0 {
			data = (*C.uint64_t)(unsafe.Pointer(&query[0]))
		}
		var cursor *C.LlevQueryCursor
		status := C.llev_transducer_query_u64(pointer, data, C.size_t(len(query)), C.size_t(maximumDistance), C.LLEV_QUERY_ORDER_TRAVERSAL, &cursor)
		runtime.KeepAlive(query)
		return cursor, check(status)
	})
}

// QueryPattern starts a dictionary × phonetic-language product query.
func (t *Transducer) QueryPattern(pattern *PhoneticPattern, maximumDistance uint8) (*Iterator, error) {
	if pattern == nil {
		return nil, errors.New("phonetic pattern is nil")
	}
	return t.withPointer(func(pointer *C.LlevTransducer) (*C.LlevQueryCursor, error) {
		pattern.mu.RLock()
		defer pattern.mu.RUnlock()
		if pattern.pointer == nil {
			return nil, errors.New("phonetic pattern is closed")
		}
		var cursor *C.LlevQueryCursor
		status := C.llev_transducer_query_pattern(pointer, pattern.pointer, C.uint8_t(maximumDistance), &cursor)
		return cursor, check(status)
	})
}

// QueryCacheStats is an immutable snapshot of aggregate TinyLFU/SIEVE policy
// counters and current bounded residency.
type QueryCacheStats struct {
	Requests        uint64
	Hits            uint64
	Misses          uint64
	Admissions      uint64
	Rejections      uint64
	Evictions       uint64
	ResidentEntries uint
	ResidentWeight  uint
}

// QueryCache is an exclusive synchronization-free cache for complete repeated
// queries. Limits apply independently to traversal and distance-then-term
// order. Create one cache per worker for parallel workloads.
type QueryCache struct {
	pointer *C.LlevQueryCache
}

// NewQueryCache retains a transducer and configures hard per-order bounds.
func NewQueryCache(transducer *Transducer, maximumEntries, maximumWeight uint) (*QueryCache, error) {
	if transducer == nil {
		return nil, errors.New("transducer is nil")
	}
	transducer.mu.RLock()
	defer transducer.mu.RUnlock()
	if transducer.pointer == nil {
		return nil, errors.New("transducer is closed")
	}
	var pointer *C.LlevQueryCache
	if err := check(C.llev_query_cache_new(transducer.pointer, C.size_t(maximumEntries), C.size_t(maximumWeight), &pointer)); err != nil {
		return nil, err
	}
	result := &QueryCache{pointer: pointer}
	runtime.SetFinalizer(result, (*QueryCache).finalize)
	return result, nil
}

func (c *QueryCache) finalize() { c.Close() }

func (c *QueryCache) requireOpen() error {
	if c == nil || c.pointer == nil {
		return errors.New("query cache is closed")
	}
	return nil
}

// Stats copies aggregate policy counters and current residency.
func (c *QueryCache) Stats() (QueryCacheStats, error) {
	if err := c.requireOpen(); err != nil {
		return QueryCacheStats{}, err
	}
	var raw C.LlevQueryCacheStats
	if err := check(C.llev_query_cache_stats(c.pointer, &raw)); err != nil {
		return QueryCacheStats{}, err
	}
	return QueryCacheStats{
		Requests: uint64(raw.requests), Hits: uint64(raw.hits),
		Misses: uint64(raw.misses), Admissions: uint64(raw.admissions),
		Rejections: uint64(raw.rejections), Evictions: uint64(raw.evictions),
		ResidentEntries: uint(raw.resident_entries), ResidentWeight: uint(raw.resident_weight),
	}, nil
}

// Clear drops resident results while preserving policy counters.
func (c *QueryCache) Clear() error {
	if err := c.requireOpen(); err != nil {
		return err
	}
	return check(C.llev_query_cache_clear(c.pointer))
}

// ResetStats resets counters while preserving residency and frequency state.
func (c *QueryCache) ResetStats() error {
	if err := c.requireOpen(); err != nil {
		return err
	}
	return check(C.llev_query_cache_reset_stats(c.pointer))
}

// Query returns an independent cursor for cached or newly computed Unicode results.
func (c *QueryCache) Query(query string, maximumDistance uint, order QueryOrder) (*Iterator, error) {
	if err := c.requireOpen(); err != nil {
		return nil, err
	}
	if !utf8.ValidString(query) {
		return nil, errors.New("query is not valid UTF-8")
	}
	input := []byte(query)
	var cursor *C.LlevQueryCursor
	status := C.llev_query_cache_query_utf8(c.pointer, (*C.char)(unsafe.Pointer(bytesPointer(input))), C.size_t(len(input)), C.size_t(maximumDistance), C.uint32_t(order), &cursor)
	runtime.KeepAlive(input)
	if err := check(status); err != nil {
		return nil, err
	}
	return newIterator(cursor), nil
}

// QueryBytes returns an independent cursor for an exact byte-domain key.
func (c *QueryCache) QueryBytes(query []byte, maximumDistance uint, order QueryOrder) (*Iterator, error) {
	if err := c.requireOpen(); err != nil {
		return nil, err
	}
	var cursor *C.LlevQueryCursor
	status := C.llev_query_cache_query_bytes(c.pointer, bytesPointer(query), C.size_t(len(query)), C.size_t(maximumDistance), C.uint32_t(order), &cursor)
	runtime.KeepAlive(query)
	if err := check(status); err != nil {
		return nil, err
	}
	return newIterator(cursor), nil
}

// QueryU64 returns an independent cursor for an exact u64-token key.
func (c *QueryCache) QueryU64(query []uint64, maximumDistance uint, order QueryOrder) (*Iterator, error) {
	if err := c.requireOpen(); err != nil {
		return nil, err
	}
	var data *C.uint64_t
	if len(query) != 0 {
		data = (*C.uint64_t)(unsafe.Pointer(&query[0]))
	}
	var cursor *C.LlevQueryCursor
	status := C.llev_query_cache_query_u64(c.pointer, data, C.size_t(len(query)), C.size_t(maximumDistance), C.uint32_t(order), &cursor)
	runtime.KeepAlive(query)
	if err := check(status); err != nil {
		return nil, err
	}
	return newIterator(cursor), nil
}

// Close releases the cache. Existing result cursors remain valid.
func (c *QueryCache) Close() {
	if c != nil && c.pointer != nil {
		C.llev_query_cache_free(c.pointer)
		c.pointer = nil
		runtime.SetFinalizer(c, nil)
	}
}

// Match owns one result copied from a bounded leased native batch. Exactly one
// of Text, Bytes, or Tokens is populated according to Domain.
type Match struct {
	Text     string
	Bytes    []byte
	Tokens   []uint64
	Distance uint
	ID       *uint64
	Domain   UnitDomain
}

// Iterator is a one-shot query cursor retaining its query-start snapshot.
type Iterator struct {
	mu      sync.Mutex
	pointer *C.LlevQueryCursor
	batch   C.LlevMatchBatchView
	index   C.size_t
	leased  bool
	ended   bool
}

func (i *Iterator) finalize() { _ = i.Close() }

// Next returns one result without materializing the remainder of the query.
func (i *Iterator) Next() (Match, bool, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.ended || i.pointer == nil {
		return Match{}, false, nil
	}
	if !i.leased || i.index == i.batch.len {
		if err := i.release(); err != nil {
			return Match{}, false, err
		}
		status := C.llev_query_cursor_next_batch(i.pointer, 256, &i.batch)
		if status == C.LLEV_STATUS_END {
			i.ended = true
			return Match{}, false, i.closeUnlocked()
		}
		if err := check(status); err != nil {
			return Match{}, false, err
		}
		i.leased = true
		i.index = 0
	}
	item := (*[1 << 30]C.LlevMatch)(unsafe.Pointer(i.batch.matches))[int(i.index)]
	i.index++
	result := Match{Distance: uint(item.distance), Domain: UnitDomain(item.unit_domain)}
	if item.has_id != 0 {
		value := uint64(item.id)
		result.ID = &value
	}
	switch result.Domain {
	case ByteDomain:
		result.Bytes = C.GoBytes(item.term_data, C.int(item.byte_len))
	case UnicodeScalarDomain:
		result.Text = C.GoStringN((*C.char)(item.term_data), C.int(item.byte_len))
	case U64Domain:
		if item.term_len != 0 {
			result.Tokens = append([]uint64(nil), unsafe.Slice((*uint64)(item.term_data), int(item.term_len))...)
		}
	default:
		return Match{}, false, fmt.Errorf("unknown match domain %d", result.Domain)
	}
	return result, true, nil
}

func (i *Iterator) release() error {
	if !i.leased {
		return nil
	}
	err := check(C.llev_query_cursor_release_batch(i.pointer, i.batch.generation))
	i.leased = false
	return err
}

// Close releases a live batch and cursor.
func (i *Iterator) Close() error { i.mu.Lock(); defer i.mu.Unlock(); return i.closeUnlocked() }
func (i *Iterator) closeUnlocked() error {
	if i.pointer == nil {
		return nil
	}
	if err := i.release(); err != nil {
		return err
	}
	err := check(C.llev_query_cursor_free(i.pointer))
	if err == nil {
		i.pointer = nil
		i.ended = true
		runtime.SetFinalizer(i, nil)
	}
	return err
}

// AutomatonSize reports compiled state and transition counts.
type AutomatonSize struct{ States, Transitions uint }

// PhoneticPattern is a reusable compiled phonetic regular language.
type PhoneticPattern struct {
	mu      sync.RWMutex
	pointer *C.LlevPhoneticPattern
}

func compilePattern(source string, llre bool) (*PhoneticPattern, error) {
	input := []byte(source)
	var pointer *C.LlevPhoneticPattern
	var status C.LlevStatus
	if llre {
		status = C.llev_phonetic_pattern_compile_llre((*C.char)(unsafe.Pointer(bytesPointer(input))), C.size_t(len(input)), &pointer)
	} else {
		status = C.llev_phonetic_pattern_compile_regex((*C.char)(unsafe.Pointer(bytesPointer(input))), C.size_t(len(input)), &pointer)
	}
	runtime.KeepAlive(input)
	if err := check(status); err != nil {
		return nil, err
	}
	result := &PhoneticPattern{pointer: pointer}
	runtime.SetFinalizer(result, (*PhoneticPattern).finalize)
	return result, nil
}

// CompilePhoneticRegex compiles the phonetic regular-expression syntax.
func CompilePhoneticRegex(source string) (*PhoneticPattern, error) {
	return compilePattern(source, false)
}

// CompilePhoneticLLRE compiles an import-free LLRE document.
func CompilePhoneticLLRE(source string) (*PhoneticPattern, error) {
	return compilePattern(source, true)
}
func (p *PhoneticPattern) finalize() { _ = p.Close() }

// Matches tests complete-string acceptance.
func (p *PhoneticPattern) Matches(text string) (bool, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.pointer == nil {
		return false, errors.New("phonetic pattern is closed")
	}
	input := []byte(text)
	var result C.uint8_t
	status := C.llev_phonetic_pattern_matches(p.pointer, (*C.char)(unsafe.Pointer(bytesPointer(input))), C.size_t(len(input)), &result)
	runtime.KeepAlive(input)
	return result != 0, check(status)
}

// Size returns compiled state and transition counts.
func (p *PhoneticPattern) Size() (AutomatonSize, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.pointer == nil {
		return AutomatonSize{}, errors.New("phonetic pattern is closed")
	}
	var states, transitions C.size_t
	err := check(C.llev_phonetic_pattern_size(p.pointer, &states, &transitions))
	return AutomatonSize{uint(states), uint(transitions)}, err
}

// Close releases a phonetic pattern; existing query cursors remain valid.
func (p *PhoneticPattern) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pointer != nil {
		C.llev_phonetic_pattern_free(p.pointer)
		p.pointer = nil
		runtime.SetFinalizer(p, nil)
	}
	return nil
}

// PhoneticRuleSet is a reusable fixed-point rewrite program.
type PhoneticRuleSet struct {
	mu      sync.RWMutex
	pointer *C.LlevPhoneticRuleSet
}

func wrapRules(pointer *C.LlevPhoneticRuleSet) *PhoneticRuleSet {
	result := &PhoneticRuleSet{pointer: pointer}
	runtime.SetFinalizer(result, (*PhoneticRuleSet).finalize)
	return result
}

// ParsePhoneticRules parses an import-free LLEV document.
func ParsePhoneticRules(source string) (*PhoneticRuleSet, error) {
	input := []byte(source)
	var pointer *C.LlevPhoneticRuleSet
	status := C.llev_phonetic_rules_parse((*C.char)(unsafe.Pointer(bytesPointer(input))), C.size_t(len(input)), &pointer)
	runtime.KeepAlive(input)
	if err := check(status); err != nil {
		return nil, err
	}
	return wrapRules(pointer), nil
}

// BuiltinPhoneticRules loads a built-in rule set.
func BuiltinPhoneticRules(kind PhoneticRuleSetKind) (*PhoneticRuleSet, error) {
	var pointer *C.LlevPhoneticRuleSet
	if err := check(C.llev_phonetic_rules_builtin(C.uint32_t(kind), &pointer)); err != nil {
		return nil, err
	}
	return wrapRules(pointer), nil
}
func (r *PhoneticRuleSet) finalize() { _ = r.Close() }

// Len returns the enabled rule count.
func (r *PhoneticRuleSet) Len() (uint, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.pointer == nil {
		return 0, errors.New("phonetic rule set is closed")
	}
	var result C.size_t
	err := check(C.llev_phonetic_rules_len(r.pointer, &result))
	return uint(result), err
}

// Apply rewrites text to a fixed point.
func (r *PhoneticRuleSet) Apply(text string) (string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.pointer == nil {
		return "", errors.New("phonetic rule set is closed")
	}
	input := []byte(text)
	var output C.LlevOwnedString
	status := C.llev_phonetic_rules_apply(r.pointer, (*C.char)(unsafe.Pointer(bytesPointer(input))), C.size_t(len(input)), &output)
	runtime.KeepAlive(input)
	if err := check(status); err != nil {
		return "", err
	}
	defer C.llev_owned_string_free(&output)
	return C.GoStringN(output.data, C.int(output.len)), nil
}

// Close releases a phonetic rule set.
func (r *PhoneticRuleSet) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.pointer != nil {
		C.llev_phonetic_rules_free(r.pointer)
		r.pointer = nil
		runtime.SetFinalizer(r, nil)
	}
	return nil
}
