// Legacy side: Maven Central 3.0.0 exactly as published (transitives and
// all), plus slf4j-nop to silence its logging. NOTHING from the vinary
// composite may appear here — the disjointness IS the experimental control.
plugins {
    java
    id("me.champeau.jmh")
}

val workloadDir = rootDir.resolve("../../workload").canonicalFile

dependencies {
    implementation(project(":common"))
    implementation("com.github.universal-automata:liblevenshtein:3.0.0")
    runtimeOnly("org.slf4j:slf4j-nop:1.7.36")
    jmh(project(":common"))
}

val benchJvmArgs = listOf(
    "-Dxl.workload=$workloadDir",
    "-Xms2g",
    "-Xmx2g",
)

jmh {
    jmhVersion = "1.37"
    benchmarkMode = listOf("avgt")
    timeUnit = "us"
    fork = providers.gradleProperty("jmh.forks").orElse("2").get().toInt()
    warmupIterations = 5
    warmup = "2s"
    iterations = providers.gradleProperty("jmh.iterations").orElse("10").get().toInt()
    timeOnIteration = "2s"
    jvmArgs = benchJvmArgs
    includes = listOf(providers.gradleProperty("jmh.includes").orElse("LegacyBench").get())
    resultFormat = "JSON"
    providers.gradleProperty("jmh.rff").orNull?.let {
        resultsFile = file(it)
    }
    providers.gradleProperty("jmh.params").orNull?.let { spec ->
        spec.split(";").forEach { pair ->
            val (key, value) = pair.split("=", limit = 2)
            benchmarkParameters.put(key, objects.listProperty(String::class.java).value(listOf(value)))
        }
    }
    // Keep diagnostic profiler selection symmetric with the vinary arm while
    // leaving the default parity run profiler-free.
    providers.gradleProperty("jmh.profilers").orNull?.let { csv ->
        profilers = csv.split(",").map(String::trim).filter(String::isNotEmpty)
    }
}

// Protobuf-beta-3 containment proof: a smoke run of the transduce path with
// -verbose:class must never load a com.google.protobuf class (the legacy
// serializer package is the only consumer). scripts/run-jvm-pair.sh executes
// this via LegacyVerifyMain + grep; the dependency stays exactly as
// published so the measured artifact is the one users run.
