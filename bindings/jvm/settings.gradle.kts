rootProject.name = "liblevenshtein"

val interopRoot = providers.gradleProperty("vinaryTreeInteropRoot")
    .orElse(providers.environmentVariable("VINARY_TREE_INTEROP_ROOT"))
    .orElse(file("../../../vinary-tree-interop").absolutePath)
val interopBuild = file("${interopRoot.get()}/bindings/jvm")

if (interopBuild.isDirectory) {
    includeBuild(interopBuild) {
        dependencySubstitution {
            substitute(module("io.vinarytree:vinary-tree-interop"))
                .using(project(":"))
        }
    }
}
