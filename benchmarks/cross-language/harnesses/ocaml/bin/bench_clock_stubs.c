/* Monotonic clock stub for the OCaml cross-language benchmark harness.
 *
 * PROTOCOL.md §9 pins every harness to a monotonic nanosecond source.
 * OCaml 5's unix library does not expose Unix.clock_gettime (verified
 * against OCaml 5.4.1), so the harness carries this minimal shim over
 * clock_gettime(CLOCK_MONOTONIC), mirroring the sanctioned Lua
 * bench_clock.so approach.
 */
#include <time.h>
#include <stdint.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

CAMLprim value bench_now_ns(value unit)
{
    CAMLparam1(unit);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    CAMLreturn(caml_copy_int64(
        (int64_t)ts.tv_sec * INT64_C(1000000000) + (int64_t)ts.tv_nsec));
}
