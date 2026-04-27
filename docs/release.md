# paku — Release Runbook

paku publishes to PyPI automatically when a `v*` tag is pushed. Releases use **PyPI Trusted Publishing** (OIDC) — no API tokens are stored in GitHub secrets.

## One-time setup (already done if PyPI listing shows the trusted publisher)

### 1. Configure PyPI Trusted Publisher

1. Sign in at https://pypi.org/manage/account/publishing/
2. Under **Add a new pending publisher**, fill in:
   - **PyPI Project Name:** `paku`
   - **Owner:** `loremcc`
   - **Repository name:** `paku`
   - **Workflow name:** `ci.yml`
   - **Environment name:** `pypi`
3. Click **Add**.

The "pending" publisher becomes a real publisher after the first successful release.

### 2. Configure GitHub environment

1. Repo → **Settings → Environments → New environment** → name it `pypi`
2. (Optional) **Required reviewers** → add yourself, so every release waits for manual approval before the publish step runs
3. (Optional) **Deployment branches** → restrict to tag pushes matching `v*`

## Release flow

1. **Bump version** in two places (must match):
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `paku/__init__.py` → `__version__ = "X.Y.Z"`

2. **Update changelog** in `tasks/changelog.md` with the release date and notable changes.

3. **Commit and tag:**
   ```bash
   git add pyproject.toml paku/__init__.py
   git commit -m "chore(release): vX.Y.Z"
   git tag vX.Y.Z
   git push origin main
   git push origin vX.Y.Z
   ```

4. **Watch CI:** the `publish` job appears only on the tag push. If you set required reviewers, approve the deployment in the GitHub Actions tab.

5. **Verify:** `pip install --upgrade paku` should pull the new version within a minute or two of publish completing.

## How the workflow works

`.github/workflows/ci.yml` has four jobs:

| Job | Triggers | Does |
|-----|----------|------|
| `lint` | every push + PR | `ruff check paku/` and `ruff format --check paku/` |
| `test` | every push + PR | `pytest` on Python 3.11 and 3.12 |
| `build` | every push + PR (after `test`) | `python -m build` → uploads `dist/` (wheel + sdist) as artifact |
| `publish` | tag push only (after `build`) | downloads `dist/` artifact, calls `pypa/gh-action-pypi-publish` with OIDC |

The `publish` job is gated on `github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')`, so feature branches and PRs never trigger it.

## Rolling back a bad release

PyPI does not allow re-uploading the same version. If you publish a broken release:

1. Yank the bad version on pypi.org (keeps installs working for pinned users, hides it from `pip install paku`).
2. Bump to the next patch version (e.g., `1.0.1` → `1.0.2`) and ship a fixed release through the normal flow.

## Troubleshooting

- **`publish` job fails with "Trusted publisher not found"**: PyPI publisher config does not match the workflow filename, environment, owner, or repo. Re-check step 1 above. The values are case-sensitive.
- **`publish` job is skipped**: you pushed a tag that doesn't start with `v`. Tags must match `v*` (e.g., `v1.0.1`, not `1.0.1` or `release-1.0.1`).
- **`build` job fails with hatchling error**: ensure `pip install build hatchling` runs before `python -m build --no-isolation`.
