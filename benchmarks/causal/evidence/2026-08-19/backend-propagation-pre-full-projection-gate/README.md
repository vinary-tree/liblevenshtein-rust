# Superseded full-projection diagnostic

These ten admitted control/treatment pairs are retained as diagnostic evidence,
not as the final propagation result. The treatment binary had SHA-256
`493c27955ef388bcd318472c55e7550bbcf9f7740635b09d62140dd1de3f1d18`.

The run exposed a provider/consumer cost mismatch: byte and character
Persistent ARTrie nodes, including the persistent vocabulary, built a dense
projection of the complete overlay on every independent query capture. The
optimized treatment therefore took roughly 111--134 microseconds per query in
those cells, versus roughly 16--43 microseconds for the lazy owned-node control.
Disabling DFS edge paging changed the affected cells by about one percent;
disabling snapshot cursors removed the regression, isolating eager full-overlay
projection as the cause.

The production fix adds
`DictionaryNode::snapshot_cursor_requires_full_projection`. Ordinary generic
query capture preserves lazy traversal while that capability is true; resource
producers such as the FFI snapshot layer can still build and amortize the dense
projection deliberately. The final matrix was restarted with the fixed binary,
rather than mixing binary identities or replacing these admitted observations.
