# Use liblevenshtein from Kotlin

The Maven artifact `io.vinarytree:liblevenshtein:4.0.0-rc.6` ships a Java API
over the stable native ABI. Kotlin consumes that exact Java API; there is no
separate Kotlin wrapper or additional Kotlin artifact. Use
`io.vinarytree:libdictenstein` to construct a dictionary resource, then pass
the resource to `Transducer`.

Compile-checked example:

```kotlin
import io.vinarytree.interop.DictionaryResource
import io.vinarytree.liblevenshtein.Algorithm
import io.vinarytree.liblevenshtein.Match
import io.vinarytree.liblevenshtein.QueryOrder
import io.vinarytree.liblevenshtein.Transducer

fun search(dictionary: DictionaryResource, input: String): List<Match> =
    Transducer(dictionary, Algorithm.STANDARD).use { transducer ->
        transducer.query(input, 2L, QueryOrder.DISTANCE_THEN_TERM).use { cursor ->
            cursor.toList()
        }
    }
```

`use` closes both native resources on normal and exceptional exits. The
transducer retains the dictionary; the cursor retains the dictionary revision
visible when `query` starts. Later producer mutations do not change that
cursor's results. `cursor.toList()` consumes it once and materializes owned
matches; for large result sets, iterate or call `forEachBatch` while respecting
its callback-scoped borrowed views. Close the cursor even after early exit.

Once RC.6 is published, its stable API reference will be the artifact's
[Javadoc](https://javadoc.io/doc/io.vinarytree/liblevenshtein/4.0.0-rc.6),
which documents the classes Kotlin calls. The
[JVM guide](README.md) covers native-access flags, packaged libraries,
phonetic operations, status handling, and dictionary collection views. The
[checked source](src/test/kotlin/io/vinarytree/liblevenshtein/KotlinUsage.kt)
is compiled by Gradle's `testClasses` task.
