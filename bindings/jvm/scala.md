# Use liblevenshtein from Scala 3

The Maven artifact `io.vinarytree:liblevenshtein:4.0.0-rc.6` exposes the
same Java API to Java, Kotlin, and Scala. No separate Scala wrapper or artifact
is published. Obtain `DictionaryResource` from a producer such as
libdictenstein; then construct a reusable transducer over it.

Compile-checked example:

```scala
import io.vinarytree.interop.DictionaryResource
import io.vinarytree.liblevenshtein.{Algorithm, Match, QueryOrder, Transducer}
import scala.jdk.CollectionConverters.*
import scala.util.Using

def search(dictionary: DictionaryResource, input: String): Vector[Match] =
  Using.resource(new Transducer(dictionary, Algorithm.STANDARD)): transducer =>
    Using.resource(transducer.query(input, 2L, QueryOrder.DISTANCE_THEN_TERM)): cursor =>
      cursor.iterator().asScala.toVector
```

`Using.resource` closes the transducer and one-shot cursor deterministically.
The cursor sees the dictionary revision captured at query creation, even if
the producer subsequently mutates. Converting to a `Vector` creates owned
match values; use the cursor incrementally for large result sets. Native
batches exposed by `forEachBatch` are borrowed only during the callback.

Once RC.6 is published, the artifact's
[Javadoc](https://javadoc.io/doc/io.vinarytree/liblevenshtein/4.0.0-rc.6)
is the normative API reference for the Java classes Scala calls. See the
[JVM guide](README.md) for native access, algorithm selection, phonetic
patterns, status handling, and dictionary collection views. The
[checked source](src/test/scala/io/vinarytree/liblevenshtein/ScalaUsage.scala)
is compiled by Gradle's `testClasses` task.
