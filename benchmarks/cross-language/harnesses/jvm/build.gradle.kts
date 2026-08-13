// Root of the Java-pair benchmark build. All configuration lives in the
// subprojects; this file only pins the shared toolchain convention.
plugins {
    id("me.champeau.jmh") version "0.7.3" apply false
}

subprojects {
    apply(plugin = "java")

    extensions.configure<JavaPluginExtension> {
        toolchain {
            languageVersion = JavaLanguageVersion.of(
                providers.gradleProperty("javaToolchain").orElse("26").get().toInt()
            )
        }
    }

    tasks.withType<JavaCompile>().configureEach {
        options.encoding = "UTF-8"
    }

    // Runtime classpath export for the launcher scripts run-one.sh generates
    // (VerifyMain / LegacyVerifyMain run outside Gradle so taskset applies to
    // a single java process, not the daemon).
    tasks.register("writeClasspath") {
        val output = project.layout.buildDirectory.file("runtime-classpath.txt")
        val runtime = project.configurations.findByName("runtimeClasspath")
        val mainOutput =
            project.extensions.getByType<SourceSetContainer>().getByName("main").output
        dependsOn(project.tasks.named("classes"))
        inputs.files(runtime)
        outputs.file(output)
        doLast {
            val entries = mutableListOf<String>()
            mainOutput.classesDirs.forEach { entries.add(it.absolutePath) }
            mainOutput.resourcesDir?.let { entries.add(it.absolutePath) }
            runtime?.resolve()?.forEach { entries.add(it.absolutePath) }
            output.get().asFile.writeText(entries.joinToString(File.pathSeparator))
        }
    }
}
