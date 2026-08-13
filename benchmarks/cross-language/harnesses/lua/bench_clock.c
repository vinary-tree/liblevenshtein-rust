/* Monotonic-clock shim for the Lua 5.4 harness (PROTOCOL.md section 9):
 * os.clock() is CPU time and MUST NOT be used, so the harness loads this
 * C module and calls bench_clock.now_ns() around every timed region. */
#define _POSIX_C_SOURCE 199309L /* clock_gettime under -std=c17 */

#include <lauxlib.h>
#include <lua.h>

#include <stdint.h>
#include <time.h>

static int now_ns(lua_State* state) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return luaL_error(state, "clock_gettime(CLOCK_MONOTONIC) failed");
    lua_pushinteger(state, (lua_Integer)((int64_t)ts.tv_sec * INT64_C(1000000000)
                                         + (int64_t)ts.tv_nsec));
    return 1;
}

int luaopen_bench_clock(lua_State* state) {
    const luaL_Reg functions[] = {{"now_ns", now_ns}, {NULL, NULL}};
    luaL_newlib(state, functions);
    return 1;
}
