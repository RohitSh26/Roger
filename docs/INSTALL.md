# Installing Roger system-wide

The one-line `pip install git+…` in the README works, but it installs Roger
into whatever Python environment happens to be active. If that's a project's
virtual environment, Roger disappears the moment you switch projects — which is
why some people end up re-running `pip install` over and over.

This guide sets Roger up **once per computer** so the `roger` command works from
any directory, in any project, forever. Pick your operating system below.

The tool for the job is **pipx**. It installs Roger into its own private
environment (so Roger's dependencies never collide with your projects) but still
puts the `roger` command on your PATH globally. One install, works everywhere.

> **Why not `sudo pip install`?** On current macOS and most Linux distributions,
> installing into the system Python is blocked on purpose (you'll see an
> `externally-managed-environment` error) because it can break tools your OS
> depends on. pipx is the supported way to install a Python command-line app
> system-wide, and it's what the Python packaging guides recommend.

You need **Python 3.10 or newer** for any of these. Check with
`python3 --version` (macOS/Linux) or `python --version` (Windows).

---

## macOS

**1. Install pipx** (once). The easiest way is Homebrew:

```bash
brew install pipx
pipx ensurepath
```

No Homebrew? Use Python's own installer instead:

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

`pipx ensurepath` adds pipx's install location to your PATH. **Close and reopen
Terminal** afterwards so the change takes effect.

**2. Install Roger:**

```bash
pipx install "git+https://github.com/RohitSh26/Roger.git"
```

To include the optional browser app (`roger app`):

```bash
pipx install "roger-cli[app] @ git+https://github.com/RohitSh26/Roger.git"
```

**3. Check it worked:**

```bash
roger --help
```

---

## Windows

Use **PowerShell** (search for it in the Start menu). Python from
[python.org](https://www.python.org/downloads/) works well — during its
installer, tick **"Add Python to PATH"**.

**1. Install pipx** (once):

```powershell
python -m pip install --user pipx
python -m pipx ensurepath
```

**Close and reopen PowerShell** afterwards so the PATH change takes effect.

**2. Install Roger:**

```powershell
pipx install "git+https://github.com/RohitSh26/Roger.git"
```

With the optional browser app (`roger app`):

```powershell
pipx install "roger-cli[app] @ git+https://github.com/RohitSh26/Roger.git"
```

**3. Check it worked:**

```powershell
roger --help
```

> **Note:** installing from GitHub needs **Git** on your machine. If you get an
> error mentioning `git`, install it from [git-scm.com](https://git-scm.com/download/win)
> (or run `winget install Git.Git`), then reopen PowerShell and try again.

---

## Linux

**1. Install pipx** (once). Most distributions package it:

```bash
# Debian / Ubuntu
sudo apt install pipx

# Fedora
sudo dnf install pipx

# Arch
sudo pacman -S python-pipx
```

If your distro doesn't have it, use Python's installer:

```bash
python3 -m pip install --user pipx
```

Then, whichever way you installed it:

```bash
pipx ensurepath
```

**Open a new terminal** afterwards so the PATH change takes effect.

**2. Install Roger:**

```bash
pipx install "git+https://github.com/RohitSh26/Roger.git"
```

With the optional browser app (`roger app`):

```bash
pipx install "roger-cli[app] @ git+https://github.com/RohitSh26/Roger.git"
```

**3. Check it worked:**

```bash
roger --help
```

> Installing from GitHub needs **Git** (`sudo apt install git`, `sudo dnf install git`, etc.).

---

## Any platform — the `uv` alternative

If you already use [uv](https://docs.astral.sh/uv/) (a fast, cross-platform
Python tool manager), it has its own global-install command and you can skip
pipx entirely:

```bash
uv tool install "git+https://github.com/RohitSh26/Roger.git"
```

With the optional browser app:

```bash
uv tool install "roger-cli[app] @ git+https://github.com/RohitSh26/Roger.git"
```

`uv` installs Roger into an isolated environment and puts `roger` on your PATH,
exactly like pipx. Upgrade with `uv tool upgrade roger-cli`; remove with
`uv tool uninstall roger-cli`.

---

## Keeping Roger up to date

```bash
pipx upgrade roger-cli
```

That re-fetches the latest version from GitHub. To update every pipx-installed
tool at once: `pipx upgrade-all`.

---

## Uninstalling

```bash
pipx uninstall roger-cli
```

---

## Working on Roger itself (editable install)

If you've cloned the repository and want your code changes to take effect
without reinstalling, install it in **editable** mode. From inside your clone:

```bash
pipx install --editable .
```

Now edits to the source are picked up the next time you run `roger`.

---

## Troubleshooting

**`roger: command not found` (or `not recognized` on Windows).**
Almost always a PATH issue. Run `pipx ensurepath` again, then **fully close and
reopen** your terminal (on Windows, reopen PowerShell). If it still isn't found,
`pipx list` will show where pipx installed it — make sure that `bin`
(macOS/Linux) or `Scripts` (Windows) directory is on your PATH.

**`externally-managed-environment` error.**
That's the error pipx exists to avoid — you're running plain `pip` against your
system Python. Use the `pipx install …` command for your OS above instead.

**`error: subprocess-exited-with-error` mentioning `git`.**
Git isn't installed or isn't on your PATH. Install Git (see your OS section
above), reopen the terminal, and retry.

**Nothing above fixed it.**
Fall back to the plain per-environment install from the README
(`pip install git+https://github.com/RohitSh26/Roger.git`) and open an issue at
<https://github.com/RohitSh26/Roger/issues> with the exact error text.

---

Once Roger is installed, head back to the [README](../README.md) for how to use
it — start by running `roger` inside any project.
