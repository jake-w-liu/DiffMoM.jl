# Aqua persistent-task shutdown policy

The repository's override was justified by a measured wrapper-loading time.
Aqua waits for loading before starting its shutdown deadline, so that
measurement did not justify changing the shutdown policy. The override and
its unsupported explanation are removed. Aqua.test_all, the existing stale
dependency exclusions, import checks and every scientific test remain.

Both supported Julia versions passed Aqua's default package checks on the
current source. A control using Aqua's documented persistent timer task was
caught by the default shutdown guard. All attempted controls and their logs
are preserved, including the earlier Event-only control that did not block
precompilation. This change introduces no repository numeric deadline.

Primary source: https://github.com/JuliaTesting/Aqua.jl/blob/v0.8.18/src/persistent_tasks.jl
and https://juliatesting.github.io/Aqua.jl/stable/persistent_tasks/.
