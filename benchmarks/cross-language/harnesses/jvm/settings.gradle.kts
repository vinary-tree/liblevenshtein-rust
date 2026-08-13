// Java-pair benchmark build: :vinary (composite-substituted local JVM
// bindings) and :legacy (Maven Central 3.0.0) with fully disjoint dependency
// graphs; :common holds the binding-free protocol core both share.
//
// The includeBuild names match the ones libdictenstein's own settings give
// these directories, so Gradle deduplicates the nested composites by root
// directory instead of erroring on a name clash.
rootProject.name = "xl-jvm-bench"

include(":common", ":vinary", ":legacy")

includeBuild("../../../../vinary-tree-interop/bindings/jvm") {
    name = "vinary-tree-interop-jvm"
    dependencySubstitution {
        substitute(module("io.vinarytree:vinary-tree-interop")).using(project(":"))
    }
}

includeBuild("../../../../bindings/jvm") {
    name = "liblevenshtein-jvm"
    dependencySubstitution {
        substitute(module("io.vinarytree:liblevenshtein")).using(project(":"))
    }
}

includeBuild("../../../../../libdictenstein/bindings/jvm") {
    name = "libdictenstein"
    dependencySubstitution {
        substitute(module("io.vinarytree:libdictenstein")).using(project(":"))
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}
