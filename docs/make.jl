using Documenter, OdeMMHPlanner

makedocs(
    sitename="OdeMMHPlanner.jl",
    modules=[OdeMMHPlanner],
    checkdocs=:exports,
    warnonly=[:missing_docs],
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Inference and Sampler Tuning" => "examples/sampling.md",
            "Optimal Control" => "examples/control.md",
        ],
        "Experiments" => "experiments.md",
        "API" => "api.md",
    ],
    remotes=nothing,
)

deploydocs(
    repo="github.com/TUM-ITR/ode-mmh-planner.git",
)