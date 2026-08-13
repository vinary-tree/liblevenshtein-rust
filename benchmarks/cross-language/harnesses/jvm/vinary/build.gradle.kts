// vinary-tree side: local composite-substituted JVM bindings + JMH.
plugins {
    java
    id("me.champeau.jmh")
}

val repoRoot: java.io.File = rootDir.resolve("../../../..").canonicalFile
val dictRoot: java.io.File = repoRoot.resolve("../libdictenstein").canonicalFile
val nativeLibraryPath =
    "${repoRoot.resolve("target/release")}:${dictRoot.resolve("target/release")}"
val workloadDir = rootDir.resolve("../../workload").canonicalFile

dependencies {
    implementation(project(":common"))
    implementation("io.vinarytree:liblevenshtein:0.10.0")
    implementation("io.vinarytree:libdictenstein:0.2.1")
    jmh(project(":common"))
}

val benchJvmArgs = listOf(
    "--enable-native-access=ALL-UNNAMED",
    "-Djava.library.path=$nativeLibraryPath",
    "-Dxl.workload=$workloadDir",
    "-Xms2g",
    "-Xmx2g",
)

jmh {
    jmhVersion = "1.37"
    benchmarkMode = listOf("avgt")
    timeUnit = "us"
    // Standard matrix profile: 2 forks x 10 x 2s (20 samples). The six
    // hypothesis-deciding cells run with -Pjmh.forks=3 -Pjmh.iterations=17
    // (51 samples, pgmcp protocol requirement).
    fork = providers.gradleProperty("jmh.forks").orElse("2").get().toInt()
    warmupIterations = 5
    warmup = "2s"
    iterations = providers.gradleProperty("jmh.iterations").orElse("10").get().toInt()
    timeOnIteration = "2s"
    jvmArgs = benchJvmArgs
    includes = listOf(providers.gradleProperty("jmh.includes").orElse("VinaryBench").get())
    resultFormat = "JSON"
    providers.gradleProperty("jmh.rff").orNull?.let {
        resultsFile = file(it)
    }
    // One cell per invocation: -Pjmh.params=algorithm=standard;distance=2;...
    providers.gradleProperty("jmh.params").orNull?.let { spec ->
        spec.split(";").forEach { pair ->
            val (key, value) = pair.split("=", limit = 2)
            benchmarkParameters.put(key, objects.listProperty(String::class.java).value(listOf(value)))
        }
    }
}
