# Pushing this to GitHub

The repository is fully initialised with one commit already made. You only
need to create the empty remote and push.

## 1. Create an empty repository on GitHub

Go to https://github.com/new and create a repository named
`dgtd-nanowire-spp`. **Do not** tick "Add a README", ".gitignore" or
"license" — this repo already has them, and initialising the remote would
force you to merge unrelated histories.

## 2. Push

```bash
cd dgtd-nanowire-spp
git remote add origin https://github.com/USERNAME/dgtd-nanowire-spp.git
git branch -M main
git push -u origin main
```

Replace `USERNAME` with your GitHub account. Use an SSH URL
(`git@github.com:USERNAME/dgtd-nanowire-spp.git`) if you have keys set up.

If you have the GitHub CLI, steps 1 and 2 collapse to:

```bash
cd dgtd-nanowire-spp
gh repo create dgtd-nanowire-spp --private --source=. --push
```

Use `--public` instead of `--private` if you want it visible.

## 3. Afterwards

- Edit the two `USERNAME` placeholders in `pyproject.toml` (`[project.urls]`).
- GitHub Actions will run the test suite automatically on the first push;
  the workflow is in `.github/workflows/tests.yml`.
- Consider adding a Zenodo hook if you want a DOI for the code when you
  publish. `CITATION.cff` is already in place.

## What is deliberately not tracked

`.gitignore` excludes simulation output — `out_*/`, `*.npz`, `*.msh`, `*.mp4`,
`*.png`, `frames/` and run logs. Meshes and result data are large and
regenerable, so they do not belong in git. If you want to version a specific
figure for the README, put it in `docs/` (that path is exempted).
