# Configured query-cache boundary sample

Run `cache_boundary.jl` from the repository root with the three local Julia
packages developed into one project and the matching native libraries selected
by `LIBLEVENSHTEIN_LIBRARY` and `LIBDICTENSTEIN_LIBRARY`.

The script warms one native TinyLFU/SIEVE cache, then records seven blocks of
5,000 warm hits after 1,000 warmup hits for each arm. Both arms create and
close one native cursor per hit. The direct arm calls the C ABI from Julia; the
facade arm uses `query(cache, "cat", 1)` and `close`. Medians are elapsed
nanoseconds per hit, including cursor ownership and Julia call overhead.

| Date and host | Direct C ABI from Julia | Julia facade | Facade minus direct |
|---|---:|---:|---:|
| 2026-10-13, Threadripper PRO 5975WX, Julia 1.13.1, Linux 7.2.9 | 3,918.0 ns | 3,570.4 ns | -347.6 ns |

This short sample verifies that the configured Julia path adds no visible
order-of-magnitude cost over a direct call on the same native cache. The small
negative difference is measurement noise or differences in the two Julia call
paths; it is not evidence that the facade is intrinsically faster. The
separate Julia boundary benchmark task owns controlled cross-host performance
budgets and uncertainty estimates.
