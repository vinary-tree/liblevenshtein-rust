using Documenter
using Liblevenshtein

makedocs(
    sitename="Liblevenshtein.jl",
    modules=[Liblevenshtein],
    build=get(ENV, "VINARY_TREE_DOC_OUTPUT", "build"),
    format=Documenter.HTML(
        edit_link=get(ENV, "VINARY_TREE_DOC_SOURCE_REF", "master"),
        repolink="https://github.com/vinary-tree/liblevenshtein-rust",
    ),
    pages=["API and usage" => "index.md"],
    checkdocs=:exports,
    repo="https://github.com/vinary-tree/liblevenshtein-rust/blob/{commit}{path}#{line}",
    warnonly=false,
)

if get(ENV, "LIBLEVENSHTEIN_DOCS_DEPLOY", "") == "1"
    deploydocs(
        repo="github.com/vinary-tree/liblevenshtein-rust.git",
        dirname="julia",
        devbranch="main",
        push_preview=true,
    )
end
