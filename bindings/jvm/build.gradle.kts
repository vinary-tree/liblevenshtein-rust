import org.gradle.api.publish.maven.MavenPublication
import org.gradle.api.tasks.bundling.Jar
import org.gradle.external.javadoc.StandardJavadocDocletOptions

plugins {
    `java-library`
    `maven-publish`
}

group = "io.vinarytree"
version = "0.10.0"

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(
            providers.gradleProperty("javaToolchain").orElse("25").get().toInt()
        )
    }
    withSourcesJar()
    withJavadocJar()
}

sourceSets.named("main") {
    resources.srcDir(layout.buildDirectory.dir("native-resources"))
}
sourceSets.named("test") {
    resources.srcDir("../conformance")
    java.srcDir("src/smoke/java")
}

tasks.withType<JavaCompile>().configureEach {
    options.release = 22
    options.encoding = "UTF-8"
}

tasks.withType<Javadoc>().configureEach {
    (options as StandardJavadocDocletOptions)
        .addStringOption("Xdoclint:all,-missing", "-quiet")
    (options as StandardJavadocDocletOptions).addBooleanOption("Werror", true)
}

repositories {
    mavenCentral()
}

dependencies {
    api("io.vinarytree:vinary-tree-interop:0.1.0")
    testImplementation(platform("org.junit:junit-bom:6.1.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty(
        "java.library.path",
        providers.gradleProperty("liblevenshtein.nativeDir").orElse("../../target/debug").get()
    )
}

val nativeResourcesByPlatform = mapOf(
    "linux-x86_64" to "META-INF/native/linux-x86_64/libliblevenshtein.so",
    "linux-aarch64" to "META-INF/native/linux-aarch64/libliblevenshtein.so",
    "macos-aarch64" to "META-INF/native/macos-aarch64/libliblevenshtein.dylib",
    "windows-x86_64" to "META-INF/native/windows-x86_64/liblevenshtein.dll",
)
val selectedNativePlatforms = providers.gradleProperty("nativePlatforms")
    .map { value -> value.split(',').map(String::trim).filter(String::isNotEmpty) }
    .orElse(nativeResourcesByPlatform.keys.toList())
val expectedNativeResources = selectedNativePlatforms.map { platforms ->
    platforms.map { platform ->
        nativeResourcesByPlatform[platform]
            ?: error("unsupported native platform requested: $platform")
    }
}

val verifyNativeResources = tasks.register("verifyNativeResources") {
    group = "verification"
    description = "Verify that the publishable JVM artifact contains every supported native library."
    inputs.files(expectedNativeResources.map { resources ->
        resources.map { layout.buildDirectory.file("native-resources/$it") }
    })
    doLast {
        val missing = expectedNativeResources.get().filterNot {
            layout.buildDirectory.file("native-resources/$it").get().asFile.isFile
        }
        check(missing.isEmpty()) {
            "missing JVM native resources: ${missing.joinToString()}"
        }
    }
}

tasks.jar {
    dependsOn(verifyNativeResources)
    manifest {
        attributes(
            "Automatic-Module-Name" to "io.vinarytree.liblevenshtein",
            "Implementation-Title" to "liblevenshtein JVM bindings",
            "Implementation-Version" to project.version,
        )
    }
}

tasks.named<Jar>("sourcesJar") {
    include("**/*.java")
    includeEmptyDirs = false
}

tasks.processResources {
    from(rootProject.file("../../LICENSE")) {
        into("META-INF")
    }
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            artifactId = "liblevenshtein"
            from(components["java"])
            pom {
                name = "liblevenshtein JVM bindings"
                description = "Snapshot-consistent streaming fuzzy search through the Java FFM API"
                url = "https://github.com/vinary-tree/liblevenshtein-rust"
                inceptionYear = "2026"
                licenses {
                    license {
                        name = "Apache License 2.0"
                        url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
                        distribution = "repo"
                    }
                }
                developers {
                    developer {
                        id = "dylon"
                        name = "Dylon Edwards"
                        email = "dylon.devo@gmail.com"
                    }
                }
                scm {
                    connection = "scm:git:https://github.com/vinary-tree/liblevenshtein-rust.git"
                    developerConnection = "scm:git:ssh://git@github.com/vinary-tree/liblevenshtein-rust.git"
                    url = "https://github.com/vinary-tree/liblevenshtein-rust"
                }
                issueManagement {
                    system = "GitHub"
                    url = "https://github.com/vinary-tree/liblevenshtein-rust/issues"
                }
            }
        }
    }
    repositories {
        maven {
            name = "staging"
            url = uri(
                providers.gradleProperty("stagingRepository")
                    .orElse(layout.buildDirectory.dir("staging-deploy").map { it.asFile.absolutePath })
                    .get()
            )
        }
    }
}
