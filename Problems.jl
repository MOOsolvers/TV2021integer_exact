#=
Problems.jl  --  the three problem classes and the one model builder.

Every problem is   min { sigma * C x : x in X }.   A class is specified by
three things, supplied by dispatch:

    objective_matrix(problem)          C, p x num_decision_vars(problem)
    to_min_sense(problem)              sigma: -1 (C x maximized) or +1 (minimized)
    add_feasible_set!(model, problem)  the variables and constraints of X

From these, ONE `build_model` produces the model `restrict_tamby` solves.  The
same method serves the reduced problems (P_Γ) of ReductionCore.jl, which supply
G = ΓC as their objective matrix and take sigma and X from the instance they
wrap.

    class                    C                        sigma   X
    MOKnapsackProblem        profits v   (p x n)      -1      {x binary : w'x <= W}
    MOAssignmentProblem      costs c     (p x n^2)    +1      assignments, x = vec(x_ij)
    MOIntegerLinearProblem   profits c   (p x n)      -1      {x >= 0 integer : a x <= b}
                                                              (C of mixed sign)

Include after TambyVanderpootenRestrict.jl, which defines `AbstractMOProblem`,
`set_exact_tolerances!` and `assert_integral_objective`.
=#

using JuMP

# ---------------------------------------------------------------------------
# What a problem class supplies
# ---------------------------------------------------------------------------

"""
    objective_matrix(problem) -> p x num_decision_vars(problem)

The objective matrix C in the original domain (profits for MOKP and ILP, costs
for MOAP).  Column j pairs with entry j of the decision vector returned by
`add_feasible_set!`.
"""
function objective_matrix end

"""
    to_min_sense(problem) -> -1.0 or +1.0

The sense sigma: the problem is `min { sigma * C x : x in X }` with
`C = objective_matrix(problem)`.

A MULTIPLIER, not a sense in JuMP's meaning -- `objective_sense(model)` returns
a `MIN_SENSE` / `MAX_SENSE` enum -- so the name says `to_min_sense` rather than
`sense` to keep the two apart.
"""
function to_min_sense end

"""
Number of decision variables: n for knapsack and ILP, n^2 for assignment.
Deliberately not `num_variables`, which JuMP exports for a `Model`.
"""
function num_decision_vars end

"""
    add_feasible_set!(model, problem) -> x

THE single definition of the feasible set X of a problem class.  Adds the
variables and constraints of X to `model` and returns the decision vector, in
the ordering `objective_matrix` assumes.

X IS A PROPERTY OF THE INSTANCE AND NEVER CHANGES.  The base solve, the reduced
solve of (P_Γ) and the range solver all optimize over the same set; they differ
only in the objective they attach and the tolerances they set.  Two callers:

  1. `build_model` below, for base and reduced problems alike (a reduced
     problem forwards to the instance it wraps; see ReductionCore.jl);
  2. `build_range_solver(base)` in ReductionCore.jl.

Solver ATTRIBUTES are deliberately NOT set here.  They belong to the caller,
which is how the two callers can disagree about tolerances -- exact gaps for
the integral objectives, zero gaps for the fractional essential components the
range solver is given -- while agreeing exactly about X.
"""
function add_feasible_set! end

# ---------------------------------------------------------------------------
# The model solved by restrict_tamby
# ---------------------------------------------------------------------------

"""
    exact_model(optimizer_factory) -> Model

An empty JuMP model with the solver settings shared by every model
`restrict_tamby` solves: one thread, silent, presolve, cuts and heuristics off,
and exact tolerances (`set_exact_tolerances!`).  The integral-objective setting
is sound because the objective data are integral: C by
`assert_integral_objective`, G = ΓC by `validated_aggregate`.
"""
function exact_model(optimizer_factory)
    model = Model(optimizer_factory)
    set_string_names_on_creation(model, false)
    set_silent(model)
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "Presolve", 0)
    set_optimizer_attribute(model, "Cuts", 0)
    set_optimizer_attribute(model, "Heuristics", 0.0)
    set_exact_tolerances!(model)
    return model
end

"""
    build_model(problem, optimizer_factory) -> (model, x, f)

The model `restrict_tamby` solves: the feasible set from `add_feasible_set!`
and the objectives `f[k] = sigma * M[k, :]' x`, with `M = objective_matrix(problem)`
and `sigma = to_min_sense(problem)`, so that every problem is a minimization.
For the maximization classes (sigma = -1) the solver's values are negated
profits; recover the profits as `-Y_N`.

`optimizer_factory` is an MOI optimizer constructor, e.g. `Gurobi.Optimizer`.
"""
function build_model(problem::AbstractMOProblem, optimizer_factory)
    model = exact_model(optimizer_factory)
    x = add_feasible_set!(model, problem)
    σ, M = to_min_sense(problem), objective_matrix(problem)
    @expression(model, f[k = 1:problem.p], sum((σ .* M[k, :]) .* x))
    return model, x, f
end

# ---------------------------------------------------------------------------
# MOKP
#
# File layout: p (line 1), n (line 2), capacity W (line 3), p profit rows of
# n values each, then one weight row of n values.
# ---------------------------------------------------------------------------

"""
    MOKnapsackProblem(filename)

A 0/1 multi-objective knapsack instance: `p` objectives, `n` items,
capacity `W`, profit matrix `v` (p x n, `v[k,j]` = profit of item `j` under
objective `k`), and weights `w` (length n).
"""
struct MOKnapsackProblem <: AbstractMOProblem
    p::Int
    n::Int
    W::Float64
    v::Matrix{Float64}
    w::Vector{Float64}
    function MOKnapsackProblem(filename::String)
        open(filename) do f
            p = parse(Int, readline(f))
            n = parse(Int, readline(f))
            W = parse(Float64, readline(f))
            v = reduce(vcat, transpose(parse.(Float64, split(readline(f)))) for i in 1:p)
            w = parse.(Float64, split(readline(f)))
            assert_integral_objective(v, "MOKnapsackProblem($filename)")
            return new(p, n, W, v, w)
        end
    end
end

"X: binary `x` with `w'x <= W`."
function add_feasible_set!(model::Model, problem::MOKnapsackProblem)
    @variable(model, x[1:problem.n], Bin)
    @constraint(model, sum(problem.w .* x) <= problem.W)
    return x
end

objective_matrix(problem::MOKnapsackProblem) = problem.v
to_min_sense(::MOKnapsackProblem) = -1.0
num_decision_vars(problem::MOKnapsackProblem) = problem.n

# ---------------------------------------------------------------------------
# MOAP (multi-objective linear assignment problem)
#
# File layout: p (line 1), n (line 2), then p blocks of n lines of n values;
# block k, line i holds c[k, i, :], the cost of assigning agent i to each task
# j under objective k.
# ---------------------------------------------------------------------------

"""
    MOAssignmentProblem(filename)

A multi-objective linear assignment instance: `p` objectives, `n`
agents/tasks, and cost tensor `c` (p x n x n, `c[k,i,j]` = cost of
assigning agent `i` to task `j` under objective `k`).
"""
struct MOAssignmentProblem <: AbstractMOProblem
    p::Int
    n::Int
    c::Array{Float64,3}
    function MOAssignmentProblem(filename::String)
        open(filename) do f
            p = parse(Int, readline(f))
            n = parse(Int, readline(f))
            c = Array{Float64,3}(undef, p, n, n)
            for k in 1:p, i in 1:n
                c[k, i, :] = parse.(Float64, split(readline(f)))
            end
            assert_integral_objective(c, "MOAssignmentProblem($filename)")
            return new(p, n, c)
        end
    end
end

"""
    add_feasible_set!(model, problem::MOAssignmentProblem) -> vec(x)

The decision vector is returned FLATTENED, column-major, so entry `(l-1)*n + i`
is agent i on task l.  That matches the rows of `objective_matrix` and the
`Vector{Float64}` that `Point.sol` and `set_start_value.` expect in
TambyVanderpootenRestrict.jl.  Any objective built against it must be flattened the same way.
"""
function add_feasible_set!(model::Model, problem::MOAssignmentProblem)
    n = problem.n
    @variable(model, x[1:n, 1:n], Bin)
    @constraint(model, [r = 1:n], sum(x[r, :]) == 1)
    @constraint(model, [l = 1:n], sum(x[:, l]) == 1)
    return vec(x)
end

"""
    objective_matrix(problem::MOAssignmentProblem) -> p x n^2

Row k is `vec(c[k, :, :])`, flattened column-major so that it pairs with the
`vec(x)` returned by `add_feasible_set!`: entry `(j-1)*n + i` is the cost of
assigning agent i to task j.
"""
function objective_matrix(problem::MOAssignmentProblem)
    p, n = problem.p, problem.n
    C = Matrix{Float64}(undef, p, n * n)
    for k in 1:p
        C[k, :] = vec(problem.c[k, :, :])
    end
    return C
end
to_min_sense(::MOAssignmentProblem) = 1.0
num_decision_vars(problem::MOAssignmentProblem) = problem.n * problem.n

# ---------------------------------------------------------------------------
# MOILP (general multi-objective integer linear program)
#
# File layout: p (line 1), n (line 2), m (line 3), then p objective rows
# (n values each), m constraint rows (n values each), and one RHS row
# (m values).
# ---------------------------------------------------------------------------

"""
    MOIntegerLinearProblem(filename)

A general multi-objective integer linear program: `p` objectives, `n`
variables, `m` `<=` constraints, objective matrix `c` (p x n), constraint
matrix `a` (m x n), and RHS `b` (length m). Feasible region is
`{x >= 0 integer : a*x <= b}`.
"""
struct MOIntegerLinearProblem <: AbstractMOProblem
    p::Int
    n::Int
    m::Int
    c::Matrix{Float64}
    a::Matrix{Float64}
    b::Vector{Float64}
    function MOIntegerLinearProblem(filename::String)
        open(filename) do f
            p = parse(Int, readline(f))
            n = parse(Int, readline(f))
            m = parse(Int, readline(f))
            c = reduce(vcat, transpose(parse.(Float64, split(readline(f)))) for i in 1:p)
            a = reduce(vcat, transpose(parse.(Float64, split(readline(f)))) for i in 1:m)
            b = parse.(Float64, split(readline(f)))
            assert_integral_objective(c, "MOIntegerLinearProblem($filename)")
            return new(p, n, m, c, a, b)
        end
    end
end

"""
    add_feasible_set!(model, problem::MOIntegerLinearProblem) -> x

X may be UNBOUNDED here, unlike the knapsack and assignment sets.  Telling
UNBOUNDED from INFEASIBLE needs Gurobi's `DualReductions` off, but that is an
ATTRIBUTE rather than a constraint, so it is set by the one caller that needs it
(`build_range_solver` in ReductionCore.jl) and not here.
"""
function add_feasible_set!(model::Model, problem::MOIntegerLinearProblem)
    @variable(model, x[1:problem.n] >= 0, Int)
    @constraint(model, [r = 1:problem.m], sum(problem.a[r, :] .* x) <= problem.b[r])
    return x
end

objective_matrix(problem::MOIntegerLinearProblem) = problem.c
to_min_sense(::MOIntegerLinearProblem) = -1.0
num_decision_vars(problem::MOIntegerLinearProblem) = problem.n
