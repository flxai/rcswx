# AGENTS.md

## Working agreement

- Treat tracked configuration, repository scripts, and nearby code as the source of truth for toolchains, commands, and conventions. Do not introduce an alternate package manager, build system, or global-tool dependency.
- Before changing code, locate the closest existing implementation and follow its structure and naming. Reuse existing scripts rather than reconstructing their commands.
- Run formatter, linter, and test commands through the repository's configured environment. Use the narrowest command that exercises the changed behavior.

## Change hygiene

- Keep each change focused on the requested behavior. Do not mix unrelated cleanup, reformatting, dependency upgrades, or generated-file updates into the diff.
- Do not edit generated, vendored, or externally managed files unless the task explicitly targets them; change their source or generation path instead.
- Preserve public behavior unless the task changes it deliberately. When an intentional interface, configuration, or workflow change affects users, update the existing documentation that describes it.
- Prefer the smallest clear implementation. Avoid new abstractions and dependencies unless the existing code demonstrates the need.

## Atomic Git history

- Before completing work, inspect the Git working tree. When changes owned by the current task form one or more semantically grouped, independently reviewable commits, stage and commit them without waiting for a separate commit request.
- Partition task-owned changes by a single user-visible behavior, bug fix, documentation change, generated-artifact set, or mechanical refactor. Keep matching tests and required lockfiles or generated outputs with their implementation.
- Do not include unrelated or pre-existing user dirt. Leave an ambiguous change uncommitted and state what decision is needed.
- Use conventional commit subjects in the form `<type>: <imperative lowercase summary>`, where `type` is one of `feat`, `fix`, `test`, `refactor`, `docs`, or `chore`.
- Before each commit, inspect the staged names and diff summary; run focused validation when practical.
- Never push, amend, reset, discard, or overwrite changes unless explicitly requested.
