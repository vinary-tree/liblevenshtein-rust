import org.gradle.api.tasks.bundling.Jar
import org.gradle.external.javadoc.StandardJavadocDocletOptions

plugins {
    `java-library`
    `maven-publish`
}

group = "io.vinarytree"
version = "0.1.0"

java {
    toolchain.languageVersion = JavaLanguageVersion.of(
        providers.gradleProperty("javaToolchain").orElse("25").get().toInt()
    )
    withSourcesJar()
    withJavadocJar()
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

tasks.named<Jar>("sourcesJar") {
    include("**/*.java")
    includeEmptyDirs = false
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            artifactId = "vinary-tree-interop"
            from(components["java"])
            pom {
                name = "vinary-tree interop ABI"
                description = "Shared retained-resource layouts for modular vinary-tree JVM bindings"
                url = "https://github.com/vinary-tree/liblevenshtein-rust/tree/main/vinary-tree-interop"
                licenses {
                    license {
                        name = "Apache License 2.0"
                        url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
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
                    url = "https://github.com/vinary-tree/liblevenshtein-rust/tree/main/vinary-tree-interop"
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
