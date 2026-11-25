#=
Julia translation of the restrict algorithm of TamVan19
(https://github.com/tambysatya/TamVan19),
Copyright (c) 2022 Satya Tamby, distributed under the BSD 3-clause licence.
Translation and modifications Copyright (c) 2026 Serpil Sayın.
See the LICENSE file for the full licence text and disclaimer.
=#

#=
TambyVanderpootenRestrict.jl

Exact multi-objective integer programming solver: a Julia port of the *restrict*
variant of the Tamby & Vanderpooten algorithm,

    S. Tamby, D. Vanderpooten (2021). Enumeration of the nondominated set of
    multiobjective discrete optimization problems. INFORMS Journal on Computing
    33(1):72-85.

following the structure of the authors' reference Haskell implementation
(https://github.com/tambysatya/TamVan19: Algorithm.hs, Zone_RA.hs,
RestrictModel.hs, Reopt.hs), with Gurobi in place of CPLEX.

Every problem is written as a MINIMIZATION of `f`.  A problem type subtypes
`AbstractMOProblem`, has a field `p` (number of objectives), and provides
`build_model(problem, optimizer_factory) -> (model, x, f)`.  The three problem
classes and `build_model` are in Problems.jl; the reduced problems used by the
objective-reduction experiments are in ReductionCore.jl and reuse
`restrict_tamby` unchanged.
=#

using JuMP
using Gurobi
using Dates

abstract type AbstractMOProblem end


# ---------------------------------------------------------------------------
# Solver tolerances
#
# `restrict_tamby` splits the search region at the optimal value of each
# single-objective subproblem, so a subproblem returned within a nonzero gap can
# silently lose a nondominated point.  Every model is therefore solved to proven
# optimality:
#
#   MIPGap = 0       disables the relative stopping test.
#   MIPGapAbs = 0.999 when the objective is integral: an incumbent within 0.999
#                    of the bound is optimal because no integer fits in between.
#                    As rigorous as a tiny gap, and cheaper, because nodes that
#                    merely tie the incumbent can still be pruned.  Pass
#                    `integral_objective = false` for fractional objective data.
#   IntFeasTol, FeasibilityTol, OptimalityTol = 1e-9, so that near-integral
#                    variables cannot move an integral objective by a unit.
# ---------------------------------------------------------------------------

"""
    set_exact_tolerances!(model; integral_objective = true)

Make the solver prove optimality instead of stopping at a relative gap.
See the comment block above for why every parameter here is set the way it
is. Called by `exact_model` in Problems.jl.
"""
function set_exact_tolerances!(model; integral_objective::Bool = true)
    set_optimizer_attribute(model, "MIPGap", 0.0)
    set_optimizer_attribute(model, "MIPGapAbs", integral_objective ? 0.999 : 1e-10)
    set_optimizer_attribute(model, "IntFeasTol", 1e-9)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    try
        set_optimizer_attribute(model, "IntegralityFocus", 1)  # Gurobi 9.0+
    catch
    end
    return model
end

"""
    assert_integral_objective(C, label)

Refuse an instance whose objective coefficients are not integers.

THE ALGORITHM, NOT JUST THE TOLERANCES, ASSUMES AN INTEGRAL OBJECTIVE.  The
fields are all `Float64`, so fractional data loads happily and then produces a
wrong answer silently.  Four places depend on integrality, and only the first
has a switch:

  1. `set_exact_tolerances!` sets `MIPGapAbs = 0.999`, sound only because an
     incumbent within 0.999 of the bound IS optimal when values are integers.
     Pass `integral_objective = false` to get 1e-10 instead.  NOTE that
     `exact_model` in Problems.jl calls it without that keyword.
  2. `restrict_tamby`'s step `δ = 1.0` turns "strictly better than `ub`" into
     "at most `ub - 1`" at `zone.ub[i] - δ`.  With fractional values every
     nondominated point in the open interval `(ub - 1, ub)` is SKIPPED, so Y_N
     comes back incomplete.  There is no correct fixed δ: any δ > 0 can skip
     points, and δ -> 0 makes the constraint non-strict and the loop
     non-terminating.  This is a change to the search logic, not a parameter.
  3. `yI = round.(ideal(...))` and `yN = round.(nadir(...))` bound the search
     region; rounding a minimum can round it UP, putting the true optimum
     outside.
  4. `Point(round.(value.(f)), round.(value.(x)))` rounds every objective vector
     and every solution vector on the way out.

So this is a hard error, not a warning: a silently incomplete Y_N that looks
ordinary is the failure mode this exists to prevent.  To support fractional
objectives, fix (2) first -- the others are mechanical.

Constraint data (`w`, `W`, `a`, `b`) is NOT checked: it does not enter the
objective, and the MIP handles it at any precision.
"""
function assert_integral_objective(C::AbstractArray{<:Real}, label::AbstractString)
    all(C .== round.(C)) || error("""
        $label has non-integer objective coefficients, which this solver cannot
        handle correctly -- it would return a silently incomplete Y_N.
        See `assert_integral_objective` for the four places that assume
        integrality and what supporting fractional data would require.""")
    return nothing
end

# ---------------------------------------------------------------------------
# Point  (mirrors Design.hs `Point`)
# ---------------------------------------------------------------------------

"An objective-space / decision-space pair. Stored and passed by reference —
never copied element-by-element — everywhere it's used below."
struct Point
    perf::Vector{Float64}
    sol::Vector{Float64}
end

Base.:(==)(a::Point, b::Point) = a.perf == b.perf

# ---------------------------------------------------------------------------
# SearchZone  (mirrors Zone_RA.hs `SearchZone`)
# ---------------------------------------------------------------------------

"""
    SearchZone(ub, lb, val, defining_points)

A search zone with upper bound `ub`. `val = (k, score)` is the
zone-selection score — the best objective `k` to restrict next, and its
score — computed ONCE at creation time by `compute_max_hv`, mirroring
Haskell's `_szVal`. It is never recomputed while the zone is alive.

`defining_points[k]` is the list of already-found points that currently
define this zone's boundary along objective `k` (mirrors
`_szDefiningPoints`).
"""
struct SearchZone
    ub::Vector{Float64}
    lb::Vector{Float64}
    val::Tuple{Int,Float64}
    defining_points::Vector{Vector{Point}}
end

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

"Coordinates of `v` excluding index `k`."
function proj(v::AbstractVector{Float64}, k::Int)
    return [v[i] for i in eachindex(v) if i != k]
end

"a strictly dominates b (a[i] < b[i] for all i)"
dominates_strict(a::Vector{Float64}, b::Vector{Float64}) = all(a[i] < b[i] for i in eachindex(a))

"a weakly dominates b (a[i] <= b[i] for all i)"
dominates_weak(a::Vector{Float64}, b::Vector{Float64}) = all(a[i] <= b[i] for i in eachindex(a))

# ---------------------------------------------------------------------------
# Selection score  (mirrors `computeHV` / `computeMaxHV`)
# ---------------------------------------------------------------------------

"log2-hypervolume-like score of `u` w.r.t. ideal point `yI`, ignoring dim k.
More negative == larger (better) hypervolume, matching the Haskell sign
convention so that `minimum` picks the best score."
function compute_hv(yI::Vector{Float64}, u::Vector{Float64}, k::Int)
    s = 0.0
    @inbounds for i in eachindex(u)
        i == k && continue
        d = u[i] - yI[i]
        s += d > 0 ? log2(d) : 0.0
    end
    return -s
end

"Best (k, score) pair for a zone with upper bound `u`. Computed once, at
zone-creation time, and cached in `SearchZone.val` — this is what turns
zone selection from an O(p^2 |U|) recomputation into an O(|U|) scan."
function compute_max_hv(yI::Vector{Float64}, u::Vector{Float64})
    p = length(u)
    best_k, best_v = 1, compute_hv(yI, u, 1)
    @inbounds for k in 2:p
        v = compute_hv(yI, u, k)
        if v < best_v
            best_k, best_v = k, v
        end
    end
    return (best_k, best_v)
end

# ---------------------------------------------------------------------------
# Zone construction & splitting (mirrors `buildInitZone`, `subdivideP`, `computeChild`)
# ---------------------------------------------------------------------------

function build_init_zone(yN::Vector{Float64}, yI::Vector{Float64})
    p = length(yN)
    ub = yN .+ 1.0
    return SearchZone(ub, copy(yI), compute_max_hv(yI, ub), [Point[] for _ in 1:p])
end

"""
    subdivide(yN, yI, pt, i, zone)

Whether splitting `zone` along direction `i`, because point `pt` was just
found, would yield a nonempty and still-useful child. Mirrors `subdivideP`.
"""
function subdivide(yN::Vector{Float64}, yI::Vector{Float64}, pt::Point, i::Int, zone::SearchZone)
    pt.perf[i] == yI[i] && return false
    p = length(yN)
    @inbounds for j in 1:p
        j == i && continue
        zone.ub[j] == yN[j] + 1 && continue  # direction j never restricted yet: always fine
        found = false
        for dp in zone.defining_points[j]
            if dp.perf[i] < pt.perf[i]
                found = true
                break
            end
        end
        found || return false
    end
    return true
end

"""
    compute_child(yI, i, zone, pt)

The child of `zone` obtained by tightening dimension `i` down to
`pt.perf[i]`. Mirrors `computeChild`.
"""
function compute_child(yI::Vector{Float64}, i::Int, zone::SearchZone, pt::Point)
    p = length(zone.ub)
    ub = copy(zone.ub)
    ub[i] = pt.perf[i]
    new_defining = Vector{Vector{Point}}(undef, p)
    @inbounds for dim in 1:p
        if dim == i
            new_defining[dim] = Point[pt]
        else
            new_defining[dim] = [dp for dp in zone.defining_points[dim] if dp.perf[i] < pt.perf[i]]
        end
    end
    return SearchZone(ub, zone.lb, compute_max_hv(yI, ub), new_defining)
end

"""
    definining_point(zone, pt) -> Union{Int,Nothing}

First direction `i` (in 1..p order) for which `pt` sits exactly on the
zone's boundary and strictly improves it elsewhere. Mirrors `definingPointP`
scanned in order, as in `updateZoneRest`.
"""
function defining_direction(zone::SearchZone, pt::Point)
    p = length(zone.ub)
    @inbounds for i in 1:p
        if pt.perf[i] == zone.ub[i] && dominates_strict(proj(pt.perf, i), proj(zone.ub, i))
            return i
        end
    end
    return nothing
end

"""
    update_search_region(yN, yI, set_a, rest, pt)

Mirrors `updateSearchRegion`: zones in `set_a` (those `pt` strictly
dominates) get split into children along every valid direction; zones in
`rest` get `pt` appended to their defining points along (at most) one
direction. Returns `(children::Vector{SearchZone}, updated_rest::Vector{SearchZone})`.
"""
function update_search_region(yN::Vector{Float64}, yI::Vector{Float64},
                               set_a::Vector{SearchZone}, rest::Vector{SearchZone}, pt::Point)
    p = length(yN)
    children = SearchZone[]
    for zi in set_a
        for i in 1:p
            if subdivide(yN, yI, pt, i, zi)
                push!(children, compute_child(yI, i, zi, pt))
            end
        end
    end
    updated_rest = Vector{SearchZone}(undef, length(rest))
    for (idx, zone) in enumerate(rest)
        i = defining_direction(zone, pt)
        if i === nothing
            updated_rest[idx] = zone
        else
            zone.defining_points[i] = vcat(Point[pt], zone.defining_points[i])
            updated_rest[idx] = zone
        end
    end
    return children, updated_rest
end

# ---------------------------------------------------------------------------
# Solved-model pruning cache  (mirrors `raSolvedModels` / `restrictSolvedPred` / `addNewModel`)
# ---------------------------------------------------------------------------

"Per-objective cache: achieved optimum value => list of projected bound
vectors that produced it. Mirrors the `Set{ModelEntry}` tree, using a Dict
keyed by the exact optimum (an O(1) average lookup instead of Haskell's
O(log n) balanced-tree lookup)."
const SolvedModels = Vector{Dict{Float64,Vector{Vector{Float64}}}}

function add_new_model!(solved::SolvedModels, k::Int, ub_proj::Vector{Float64}, yk::Float64)
    bucket = get!(() -> Vector{Vector{Float64}}[], solved[k], yk)
    push!(bucket, ub_proj)
    return nothing
end

"""
    restrict_solved_pred(solved, zone) -> Bool

`true` if `zone` is *not* already known-redundant given previously solved
models, i.e. it should be kept for exploration. Mirrors `restrictSolvedPred`.
"""
function restrict_solved_pred(solved::SolvedModels, zone::SearchZone)
    p = length(zone.ub)
    @inbounds for k in 1:p
        uk = zone.ub[k]
        haskey(solved[k], uk) || continue
        zone_proj = proj(zone.ub, k)
        for prev_ub in solved[k][uk]
            # zone_proj and prev_ub are both already projected (index k
            # dropped on both sides), so compare them elementwise.
            if dominates_weak(zone_proj, prev_ub)
                return false
            end
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Infeasibility pruning  (mirrors `onProjOptFailure` / `restrictPred`)
# ---------------------------------------------------------------------------

"Remove every zone whose bound (projected away from k) is at least as tight
as `ub`'s — they're guaranteed infeasible too, since `ub` already was.
Mirrors `onProjOptFailure` / `restrictPred`. Returns a new vector; does not
mutate `zones` in place."
function drop_dominated_infeasible(zones::Vector{SearchZone}, k::Int, ub::Vector{Float64})
    keep = SearchZone[]
    ub_proj = proj(ub, k)
    for z in zones
        # Both sides are already projected (index k dropped).
        if dominates_weak(proj(z.ub, k), ub_proj)
            # z.ub is at least as restrictive as ub in every non-k coordinate: also infeasible.
            continue
        end
        push!(keep, z)
    end
    return keep
end

# ---------------------------------------------------------------------------
# Stats  (mirrors Stats.hs `RestrictStats`, trimmed to what's easy to reuse)
# ---------------------------------------------------------------------------

mutable struct RestrictTambyStats
    n_models_solved::Int
    n_infeasible::Int
    n_redundant::Int
    n_lucky::Int
    l_max::Int
    l_total::Int
end
RestrictTambyStats() = RestrictTambyStats(0, 0, 0, 0, 0, 0)

# ---------------------------------------------------------------------------
# Main driver  (mirrors `restrictAlgo` / `restrictBuildNDPT`)
# ---------------------------------------------------------------------------

"""
    restrict_tamby(problem, solver = Gurobi.Optimizer, δ = 1.0;
                   time_limit_ms = Inf, iter_limit = Inf, prune = true, verbose = false)

Enumerate the nondominated set of `problem` (minimization of `f`).

Returns `(Y_N, X_E, U_history, n_iter, n_infeasible, stats)`: the nondominated
points, one efficient solution per point, the number of open search zones at
each iteration, the iteration count, the number of infeasible subproblems, and a
`RestrictTambyStats`.  `δ = 1` is the step for integral objectives.  The run is
complete iff it stopped because no search zone remained, i.e. neither
`time_limit_ms` nor `iter_limit` was reached.  `prune = true` enables the
solved-model cache of the reference implementation (`raSolvedModels`).
"""
function restrict_tamby(problem::AbstractMOProblem, solver = Gurobi.Optimizer, δ::Float64 = 1.0;
                         time_limit_ms = Inf, iter_limit = Inf,
                         prune::Bool = true, verbose::Bool = false)
    p = problem.p
    model, x, f = build_model(problem, solver)
    yI = round.(ideal(model, f))
    yN = round.(nadir(model, f))
    yN = fill(maximum(yN), p)
    verbose && @info "yI = $yI, yN = $yN"

    # Persistent objective-bound constraints, RHS mutated in place every
    # iteration instead of being added/deleted (see file header, point 3).
    @constraint(model, obj_ctrs[i = 1:p], f[i] <= yN[i])

    zones = SearchZone[build_init_zone(yN, yI)]
    solved = SolvedModels([Dict{Float64,Vector{Vector{Float64}}}() for _ in 1:p])
    solutions = Dict{Vector{Float64},Vector{Float64}}()

    U_history = Int[]
    n_iter = 0
    stats = RestrictTambyStats()
    start_time = now()

    # Run the LP with objective `obj_idx` favoured (rhs[obj_idx] = yN[obj_idx],
    # the rest bounded by `bound(i)`), optionally warm-started, and return a
    # Point or `nothing` if infeasible.
    function solve_with(obj_idx::Int, bound::Function, warm_start::Union{Nothing,Vector{Float64}})
        for i in 1:p
            set_normalized_rhs(obj_ctrs[i], i == obj_idx ? yN[i] : bound(i))
        end
        @objective(model, Min, f[obj_idx])
        if warm_start !== nothing
            set_start_value.(x, warm_start)
        end
        optimize!(model)
        termination_status(model) == OPTIMAL || return nothing
        return Point(round.(value.(f)), round.(value.(x)))
    end

    # Lexicographic cleanup: minimize sum(f) s.t. f .<= pt.perf, warm-started
    # at pt.sol. Mirrors `reopt`. Falls back to `pt` if this solve fails,
    # exactly like the Haskell code.
    function reopt(pt::Point)
        for i in 1:p
            set_normalized_rhs(obj_ctrs[i], pt.perf[i])
        end
        @objective(model, Min, sum(f))
        set_start_value.(x, pt.sol)
        optimize!(model)
        termination_status(model) == OPTIMAL || return pt
        return Point(round.(value.(f)), round.(value.(x)))
    end

    while !isempty(zones) && ((now() - start_time).value < time_limit_ms) && (n_iter < iter_limit)
        push!(U_history, length(zones))
        n_iter += 1
        stats.l_max = max(stats.l_max, length(zones))
        stats.l_total += length(zones)

        # NOTE: argmin(f, itr) returns the *element* minimizing f, not its
        # index, so we scan indices explicitly to be able to build `zrest`.
        idx = argmin(i -> zones[i].val[2], eachindex(zones))
        zone = zones[idx]
        k = zone.val[1]
        zrest = [zones[j] for j in eachindex(zones) if j != idx]

        verbose && @info "iter $n_iter: |U|=$(length(zones)) k=$k ub=$(zone.ub)"

        defpts = zone.defining_points[k]
        warm_start = if !isempty(solutions) && zone.ub[k] != yN[k] + 1 && !isempty(defpts)
            best = argmin(dp -> sum(dp.perf), defpts)   # argmin(f, itr) returns the element itself
            best.sol
        else
            nothing
        end

        pt = solve_with(k, i -> zone.ub[i] - δ, warm_start)
        stats.n_models_solved += 1

        if pt === nothing
            stats.n_infeasible += 1
            zones = drop_dominated_infeasible(zrest, k, zone.ub)
            continue
        end

        if pt.perf[k] < zone.ub[k]
            # pt genuinely belongs to this zone: clean it up, split this
            # zone plus whichever remaining zones pt now dominates.
            pt2 = reopt(pt)
            set_a = SearchZone[z for z in zrest if dominates_strict(pt2.perf, z.ub)]
            rest = SearchZone[z for z in zrest if !dominates_strict(pt2.perf, z.ub)]
            pushfirst!(set_a, zone)

            children, updated_rest = update_search_region(yN, yI, set_a, rest, pt2)
            add_new_model!(solved, k, proj(zone.ub, k), pt2.perf[k])
            if prune
                children = filter(z -> restrict_solved_pred(solved, z), children)
            end
            zones = vcat(updated_rest, children)
            get!(solutions, pt2.perf, pt2.sol)
        else
            # pt is outside the zone we asked for: redundant, unless it
            # turns out none of the zone's current defining points already
            # dominate it ("lucky point").
            lucky = !any(dp -> dominates_weak(dp.perf, pt.perf), defpts)
            if lucky
                stats.n_lucky += 1
                pt2 = reopt(pt)
                set_a = SearchZone[z for z in zrest if dominates_strict(pt2.perf, z.ub)]
                rest = SearchZone[z for z in zrest if !dominates_strict(pt2.perf, z.ub)]

                children, updated_rest = update_search_region(yN, yI, set_a, rest, pt2)
                add_new_model!(solved, k, proj(zone.ub, k), pt2.perf[k])
                if prune
                    children = filter(z -> restrict_solved_pred(solved, z), children)
                end
                zones = vcat(updated_rest, children)
                get!(solutions, pt2.perf, pt2.sol)
            else
                stats.n_redundant += 1
                add_new_model!(solved, k, proj(zone.ub, k), pt.perf[k])
                zones = zrest
            end
        end
    end

    Y_N = collect(keys(solutions))
    X_E = collect(values(solutions))
    return Y_N, X_E, U_history, n_iter, stats.n_infeasible, stats
end


"Ideal point: independently minimize each f[k] over the feasible region."
function ideal(model, f)
    p = length(f)
    yI = zeros(p)
    for k in 1:p
        @objective(model, Min, f[k])
        optimize!(model)
        yI[k] = objective_value(model)
    end
    return yI
end

"Nadir point (loose/safe upper bound): independently maximize each f[k]
over the feasible region. `restrict_tamby` immediately homogenizes this to
a single uniform value (`fill(maximum(yN), p)`), so precision here matters
less than validity as an upper bound."
function nadir(model, f)
    p = length(f)
    yN = zeros(p)
    for k in 1:p
        @objective(model, Max, f[k])
        optimize!(model)
        yN[k] = objective_value(model)
    end
    return yN
end
