# Vinary Tree liblevenshtein for Ruby

This gem supports maintained Ruby 3.3 through the latest Ruby release. Queries
are one-shot `Enumerable` objects backed by a native cursor; only a bounded
batch is leased and each yielded `Match` owns its term. A query captures its
dictionary revision at construction and can outlive the source dictionary.

Any modular dictionary gem can implement `with_resource { |context, vtable| }
to participate in O(1) retained-resource handoff. The libdictenstein gem does
so without serialization or an object-format conversion.

Set `LIBLEVENSHTEIN_LIBRARY` for a source-tree build. Release gems contain the
platform shared library under `lib/vinary_tree/native/<platform>/`; a system
installation remains a supported loader fallback.
