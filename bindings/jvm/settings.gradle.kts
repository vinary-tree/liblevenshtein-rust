rootProject.name = "liblevenshtein"

includeBuild("../../vinary-tree-interop/bindings/jvm") {
    dependencySubstitution {
        substitute(module("io.vinarytree:vinary-tree-interop"))
            .using(project(":"))
    }
}
