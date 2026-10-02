using Documenter
using Liblevenshtein

const DOCS_ROOT = @__DIR__
const DOCS_BUILD = get(ENV, "VINARY_TREE_DOC_OUTPUT", "build")
const DOCS_DEPLOY = get(ENV, "LIBLEVENSHTEIN_DOCS_DEPLOY", "") == "1"

# Documenter copies its build target relative to the same root when deploying.
DOCS_DEPLOY && DOCS_BUILD != "build" &&
    error("Julia docs deployment requires the docs-root build target")

makedocs(
    root=DOCS_ROOT,
    sitename="Liblevenshtein.jl",
    modules=[Liblevenshtein],
    build=DOCS_BUILD,
    format=Documenter.HTML(
        edit_link=get(ENV, "VINARY_TREE_DOC_SOURCE_REF", "master"),
        repolink="https://github.com/vinary-tree/liblevenshtein-rust",
    ),
    pages=["API and usage" => "index.md"],
    checkdocs=:exports,
    repo="https://github.com/vinary-tree/liblevenshtein-rust/blob/{commit}{path}#{line}",
    warnonly=false,
)

isfile(joinpath(DOCS_ROOT, DOCS_BUILD, "index.html")) ||
    error("Documenter did not generate the Julia guide")

if DOCS_DEPLOY
    deploydocs(
        root=DOCS_ROOT,
        target="build",
        repo="github.com/vinary-tree/liblevenshtein-rust.git",
        dirname="julia",
        devbranch="master",
        push_preview=false,
    )
end
