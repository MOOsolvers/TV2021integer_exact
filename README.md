# Tamby–Vanderpooten (restrict) in Julia
 
A Julia/JuMP implementation of the exact algorithm of Tamby and Vanderpooten for enumerating the full nondominated set of a multi-objective integer program

> S. Tamby, D. Vanderpooten (2021). *Enumeration of the nondominated set of
> multiobjective discrete optimization problems.* INFORMS Journal on Computing
> 33(1):72–85.

Please cite their paper when using this code.

## Origin of this code

This code is a **translation into Julia, made by Claude (Anthropic's AI model), of the
original Haskell implementation shared by the authors**:

<https://github.com/tambysatya/TamVan19>

It follows the *restrict* variant of that implementation module by module
(`Algorithm.hs`, `Zone_RA.hs`, `RestrictModel.hs`, `Reopt.hs`, `Design.hs`), and
the comments in the source point to the Haskell function each Julia function
mirrors (`SearchZone`, `computeMaxHV`, `buildInitZone`, `subdivideP`,
`computeChild`, `raSolvedModels`, `restrictSolvedPred`, `addNewModel`, `reopt`).
The original repository states that it was updated for CPLEX 22.10 and is not
the exact code used for the paper's experiments, which is linked from the paper
(<https://www.lamsade.dauphine.fr/~vdp/tv19>). This translation follows the
GitHub version, two-stage configuration (`twostage`: each subproblem is
followed by the lexicographic re-optimization `reopt`).

The main differences from the original are:

* **Gurobi instead of CPLEX**, through JuMP.
* Only the two-stage approach is implemented; the original's direct approach
  (`scal`) is not.
* Solved-model cache stored as a `Dict` keyed by the optimal value instead of a
  balanced tree (same pruning, O(1) average lookup).
* Objective-bound constraints are created once and their right-hand sides are
  updated in place at each iteration, instead of being added and removed.
* An explicit check (`assert_integral_objective`) refuses instances with
  non-integer objective coefficients. The algorithm (step δ = 1, rounded ideal and
  nadir points, rounded solutions) assumes integral objectives, and fractional
  data would otherwise return an incomplete nondominated set without any warning.
* Instance files are plain whitespace-separated text, not the Haskell list
  syntax of the original `.dat` files.

### Solver settings

| setting | original (CPLEX) | this code (Gurobi) |
|---|---|---|
| threads | 1 (`Threads = 1`) | 1 (`Threads = 1`) |
| relative / absolute MIP gap | 0 / 0 (`EpGap`, `EpAGap`) | 0 / 0.999 (`MIPGap`, `MIPGapAbs`) |
| feasibility, integrality, optimality tolerances | CPLEX defaults | 1e-9 |
| MIP start from a known solution | yes (`AdvInd = 1`) | yes (`set_start_value`) |
| presolve | CPLEX default (on) | off (`Presolve = 0`) |
| cutting planes | CPLEX default (on) | off (`Cuts = 0`) |
| primal heuristics | CPLEX default (on) | off (`Heuristics = 0`) |

Both versions solve every subproblem to proven optimality: an absolute gap of
0.999 is exact when the objective is integral, because no integer lies strictly
between the incumbent and the bound. Turning off presolve, cuts and heuristics
does not affect the nondominated set returned, only the solution times.

Any error in the translation is ours, not the original authors'.

## Debugging, testing and verification

### Verification against published reference sets

The translation was verified on **160 multi-objective 0/1 knapsack instances**
from the Multiobjective Optimization Library of Kirlik and Sayın (2014)
(<http://home.ku.edu.tr/~moolibrary/>), for which the exact nondominated sets
are available. Every instance was solved to completion and
the computed set compared point by point with the published one:

| p | n | instances | \|Y_N\| (min – max) | mean CPU (s) | missing | extra | dominated |
|---|---|---|---|---|---|---|---|
| 3 | 10, 20, …, 100 | 100 | 5 – 10 179 | 40.6 | 0 | 0 | 0 |
| 4 | 10, 20, 30, 40 | 40 | 5 – 5 018 | 12.7 | 0 | 0 | 0 |
| 5 | 10, 20 | 20 | 6 – 426 | 1.3 | 0 | 0 | 0 |
| **total** | | **160** | | | **0** | **0** | **0** |

For each instance three checks were made: (1) every published point is found
and no point outside the published set is returned; (2) no returned point
dominates another (a check that does not depend on the reference set); (3) the
run ended with no open search zone, i.e. it was not truncated.
**All 160 sets coincide exactly with the published reference sets.**

### Comparison with the results published by Tamby and Vanderpooten

No reference nondominated sets are published for the MOKP and MOAP instances
used by Tamby and Vanderpooten (2021) (available at
<https://www.lamsade.dauphine.fr/~vdp/tv19>). Tables 1 and 2 of their paper
report averages over 10 instances per class, so this implementation was run on
the same 70 instances and its averages are compared with theirs below. Each
cell gives **this implementation / the paper's TwoStage variant**, the variant
that `restrict` followed by `reopt` corresponds to (Direct in brackets where it
differs).

**MOKP** (paper, Table 1)

| p | n | \|Y_N\| (mean) | #Iterations (mean) | #Infeasible (mean) | \|U\| max (mean) |
|---|---|---|---|---|---|
| 3 | 100 | 4 090.0 / 4 090.0 | 7 448.5 / 7 446.9 (7 358.8) | 0 / 0 | 382.0 / 377.9 (403.9) |
| 4 | 50 | 2 240.1 / 2 240.1 | 10 335.0 / 10 333.2 (10 280.7) | 0 / 0 | 3 351.4 / 3 348.4 (3 396.6) |
| 5 | 50 | 8 788.3 / 8 788.3 | 122 056.7 / 122 051.8 (121 795.4) | 0 / 0 | 54 118.7 / 54 101.0 (54 501.2) |
| 6 | 30 | 1 710.4 / 1 710.4 | 61 104.4 / 61 103.0 (61 075.1) | 0 / 0 | 38 907.5 / 38 909.5 (39 030.9) |

**MOAP** (paper, Table 2)

| p | n × n | \|Y_N\| (mean) | #Iterations (mean) | #Infeasible (mean) | \|U\| max (mean) |
|---|---|---|---|---|---|
| 2 | 100 × 100 | 202.8 / 202.8 | 203.8 / 203.8 | 0 / 0 | 1.0 / 1.0 |
| 3 | 50 × 50 | 16 867.2 / 16 867.2 | 25 288.9 / 25 305.2 (23 538.1) | 0 / 0 | 403.2 / 375.9 (524.3) |
| 4 | 20 × 20 | 24 750.0 / 24 750.0 | 64 318.6 / 64 406.3 (62 027.1) | 0 / 0 | 8 292.7 / 7 999.6 (9 331.4) |

**The mean size of the nondominated set matches the published value exactly
in all seven classes.** Because these are averages over 10 instances, this is
strong evidence but not a point-by-point proof that every set is complete.
As in the paper, no infeasible subproblem is ever solved. The number of
iterations and the largest search region agree with the TwoStage figures to
within 0.1 % on MOKP and 4 % on MOAP. Small differences are expected: when a
subproblem has several optimal solutions, Gurobi and CPLEX can return
different ones, which changes the order in which points are found but not the
final set.

Mean CPU time per instance (seconds). Not directly comparable: different
machine, Gurobi instead of CPLEX, and presolve, cuts and heuristics turned off
(see *Solver settings*). This code uses one thread, as does the authors' GitHub version.

| class | this implementation | paper, TwoStage |
|---|---|---|
| MOKP p = 3, n = 100 | 106.0 | 273.5 |
| MOKP p = 4, n = 50 | 48.4 | 210.7 |
| MOKP p = 5, n = 50 | 1 975.3 | 4 298.0 |
| MOKP p = 6, n = 30 | 738.5 | 1 380.3 |
| MOAP p = 2, 100 × 100 | 38.1 | 62.5 |
| MOAP p = 3, 50 × 50 | 2 222.2 | 4 246.2 |
| MOAP p = 4, 20 × 20 | 1 073.9 | 4 789.8 |

## Requirements

* Julia ≥ 1.10
* JuMP
* Gurobi with a valid licence (Gurobi.jl)

## Problems that can be modelled

`restrict_tamby` accepts any problem type that subtypes `AbstractMOProblem` and
provides `build_model`, so any multi-objective integer program with integral
objective coefficients can be solved. `Problems.jl` provides three classes, each
with a reader for its instance file:

| type | problem | variables | feasible set |
|---|---|---|---|
| `MOKnapsackProblem` | max p profit functions v_k'x | x ∈ {0,1}ⁿ | w'x ≤ W |
| `MOAssignmentProblem` | min p cost functions Σ c_kij x_ij | x ∈ {0,1}^{n×n} | each agent assigned to exactly one task and each task to exactly one agent |
| `MOIntegerLinearProblem` | max p linear functions c_k'x (coefficients of any sign) | x ∈ ℤⁿ, x ≥ 0 | Ax ≤ b (m constraints) |

`MOIntegerLinearProblem` covers general (not only binary) integer variables.
The feasible set must be bounded, or at least each objective must be bounded
over it, since the algorithm starts from the ideal and nadir points. The
knapsack and assignment classes are special cases, but their own readers and
models are kept because they match the published instance files.

Instance files are plain text, one number per value separated by spaces, one
row per line:

| type | file layout |
|---|---|
| MOKP | p; n; W; p lines of n profits; 1 line of n weights |
| MOAP | p; n; then p blocks of n lines of n costs (block k, line i = costs of agent i under objective k) |
| ILP | p; n; m; p lines of n objective coefficients; m lines of n constraint coefficients; 1 line of m right-hand sides |

Maximization classes are passed to the solver as minimization of −Cx, so their
`Y_N` holds negated values; the original values are `-Y_N`.

New classes are added by defining `objective_matrix`, `to_min_sense` and
`add_feasible_set!` for the new type; the one `build_model` then builds the
model. The original Haskell code reads knapsack (`KP`), assignment (`AP`) and an
experimental generic binary format (`MODO`, l ≤ Ax ≤ u, x ∈ {0,1}ⁿ).

## Usage

A problem type subtypes `AbstractMOProblem`, has a field `p` (number of
objectives) and provides

```julia
build_model(problem, optimizer_factory) -> (model, x, f)
```

where `f` is the vector of the `p` objective expressions, all to be
**minimized** (maximization problems are passed as `-c'x`). Then

```julia
include("TambyVanderpootenRestrict.jl")
include("Problems.jl")        # MOKnapsackProblem, MOAssignmentProblem, MOIntegerLinearProblem

problem = MOKnapsackProblem("MOKP_p-3_n-100_1.txt")

Y_N, X_E, U_history, n_iter, n_infeasible, stats =
    restrict_tamby(problem, Gurobi.Optimizer;
                   time_limit_ms = Inf, iter_limit = Inf,
                   prune = true, verbose = false)
```

| output | meaning |
|---|---|
| `Y_N` | nondominated points (objective vectors) |
| `X_E` | one efficient solution per point |
| `U_history` | number of open search zones at each iteration |
| `n_iter` | number of iterations (single-objective subproblems solved) |
| `n_infeasible` | number of infeasible subproblems |
| `stats` | `n_models_solved`, `n_redundant`, `n_lucky`, `l_max` (largest number of open zones) |

The enumeration is complete if and only if the run stopped because no search
zone remained, i.e. neither `time_limit_ms` nor `iter_limit` was reached.
`prune = true` enables the solved-model cache of the reference implementation.

## Limitations

* Objective coefficients must be integers (enforced; see `assert_integral_objective`
  for the four places in the algorithm that depend on it).
* Only Gurobi has been used and tested.
* The verification and comparisons above cover the knapsack and assignment
  classes; the general integer class (`MOIntegerLinearProblem`) has not been
  checked against published nondominated sets.

## Credits

Algorithm and original Haskell implementation: Satya Tamby and Daniel
Vanderpooten. Please cite their paper when using this code.

The knapsack instances and the code that generated the reference nondominated sets used for verification are from the Multiobjective Optimization Library https://github.com/srplsyn/MOOLibrary accompanying

> G. Kirlik, S. Sayın (2014). A new algorithm for generating all nondominated
> solutions of multiobjective discrete optimization problems. *European Journal
> of Operational Research* 232(3):479–488.

Julia translation: Claude (Anthropic). Debugging, testing and verification:
Serpil Sayın.

## Licence

BSD 3-clause; see `LICENSE`. The translated code is derived from TamVan19,
Copyright (c) 2022 Satya Tamby, whose copyright notice, conditions and
disclaimer are reproduced there unchanged.
