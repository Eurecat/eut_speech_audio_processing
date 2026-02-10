# Pre-commit Hook Guide

This document provides comprehensive information about using pre-commit hooks with Ruff for automatic Python code formatting in this repository.

## Table of Contents

- [What is Pre-commit?](#what-is-pre-commit)
- [Installation](#installation)
  - [Ubuntu/Linux CLI Setup](#ubuntulinux-cli-setup)
  - [VS Code Setup](#vs-code-setup)
- [How It Works](#how-it-works)
- [Usage Examples](#usage-examples)
- [Common Ruff Errors](#common-ruff-errors-and-how-to-fix-them)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## What is Pre-commit?

Pre-commit is a framework that manages Git hooks to run automated checks before each commit. In this repository, we use it to automatically format and lint Python code with Ruff.

## Installation

### Ubuntu/Linux CLI Setup

**Step 1: Install Required Tools**

```bash
# Install Python and pip (if not already installed)
sudo apt update
sudo apt install python3 python3-pip -y

# Install pre-commit
pip install pre-commit

# Install ruff (optional, pre-commit will manage it automatically)
pip install ruff
```

**Step 2: Configure Pre-commit Hooks**

From the repository root:

```bash
# Install the git hooks
pre-commit install
```

Now, every time you commit, Ruff will automatically:
- ✅ Format your Python code
- ✅ Fix auto-fixable linting issues
- ⚠️ Stop the commit if there are issues that need manual attention

### VS Code Setup

**Step 1: Install Ruff Extension**

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Ruff" by Astral Software
4. Install the extension

**Step 2: Configure VS Code Settings**

Add to your `.vscode/settings.json`:

```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "ruff.lint.args": ["--config=pyproject.toml"],
  "ruff.format.args": ["--config=pyproject.toml"]
}
```

This will:
- Format your code automatically when you save
- Fix linting issues on save
- Organize imports automatically

**Step 3: Install Pre-commit Hooks (Optional)**

Even with VS Code, you can still install pre-commit hooks as a safety net:

```bash
pip install pre-commit
pre-commit install
```

### Manual Formatting

You can run Ruff manually at any time:

```bash
# Format all Python files in src/
ruff format src/

# Check and fix linting issues
ruff check src/ --fix

# Check without fixing (dry-run)
ruff check src/

# Run on specific file
ruff format src/my_module/my_file.py
ruff check src/my_module/my_file.py --fix
```

## How It Works

### Normal Workflow

When you commit code, the pre-commit hook automatically runs:

1. You run `git commit`
2. Pre-commit intercepts the commit
3. Ruff checks and formats all Python files in `src/`
4. **If files are modified**: commit is blocked → review changes → stage them → commit again
5. **If errors need manual fixing**: commit is blocked → fix errors → stage fixes → commit again
6. **If everything is clean**: commit succeeds immediately ✅

### What Happens During a Commit

```bash
# You make changes and stage them
git add src/my_file.py

# You try to commit
git commit -m "Add new feature"

# Pre-commit runs automatically
# → Ruff formats your code
# → If changes were made, commit is BLOCKED

# Review the changes
git diff src/my_file.py

# If you agree with the changes, stage them
git add src/my_file.py

# Commit again
git commit -m "Add new feature"
# ✅ Success!
```

### If Errors Can't Be Auto-Fixed

Pre-commit will show specific errors (e.g., lines too long, naming conventions):

```bash
git commit -m "Add new feature"

# ⚠️ Ruff shows errors:
# E501 Line too long (120 > 100)
# N801 Class name should use CapWords

# Fix the errors in your editor
# Then stage and commit again
git add src/my_file.py
git commit -m "Add new feature"
```

### Emergency Bypass (NOT Recommended)

In rare emergencies, you can skip pre-commit:

```bash
git commit -m "Emergency fix" --no-verify
```

⚠️ **Use only when absolutely necessary!** Fix formatting issues as soon as possible.

## Usage Examples

### Scenario 1: Clean Commit (No Issues)

When your code is already properly formatted:

```bash
$ git add src/my_module/my_file.py
$ git commit -m "Add new feature"

ruff.....................................................................Passed
ruff-format..............................................................Passed
[jazzy-devel abc1234] Add new feature
 1 file changed, 10 insertions(+)
```

✅ Commit succeeds immediately!

### Scenario 2: Auto-fixable Issues

When Ruff can automatically fix formatting issues:

```bash
$ git add src/my_module/my_file.py
$ git commit -m "Add new feature"

ruff.....................................................................Passed
ruff-format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook

1 file reformatted

# ⚠️ Commit blocked! Files were auto-formatted
```

**What to do:**

```bash
# Review the changes
$ git diff src/my_module/my_file.py

# If you agree with the formatting, stage the changes
$ git add src/my_module/my_file.py

# Commit again
$ git commit -m "Add new feature"

ruff.....................................................................Passed
ruff-format..............................................................Passed
[jazzy-devel abc1234] Add new feature
 1 file changed, 10 insertions(+)
```

✅ Commit succeeds!

### Scenario 3: Manual Fixes Required

When there are linting issues that need manual attention:

```bash
$ git add src/my_module/my_file.py
$ git commit -m "Add new feature"

ruff.....................................................................Failed
- hook id: ruff
- exit code: 1

src/my_module/my_file.py:10:101: E501 Line too long (120 > 100)
   |
 8 | def my_function():
 9 |     # This is a very long comment that exceeds the maximum line length of 100 characters and needs to be split
10 |     very_long_variable_name = "This is a very long string that exceeds the maximum line length of 100 characters"
   |                                                                                                     ^^^^^^^^^^^^^^^^^^^^ E501
11 |     return very_long_variable_name
   |

src/my_module/my_file.py:15:7: N801 Class name `myClass` should use CapWords convention
   |
13 | 
14 | # Bad class name
15 | class myClass:
   |       ^^^^^^^ N801
16 |     pass
   |

Found 2 errors.

# ⚠️ Commit blocked! Manual fixes needed
```

**What to do:**

```bash
# Fix the issues in your editor
# 1. Break the long line into multiple lines
# 2. Rename myClass to MyClass

# Stage the fixes
$ git add src/my_module/my_file.py

# Commit again
$ git commit -m "Add new feature"

ruff.....................................................................Passed
ruff-format..............................................................Passed
[jazzy-devel abc1234] Add new feature
 1 file changed, 10 insertions(+)
```

✅ Commit succeeds!

### Scenario 4: Bypassing Pre-commit (Emergency Only)

In rare emergencies, you can skip pre-commit hooks:

```bash
$ git commit -m "Emergency hotfix" --no-verify
[jazzy-devel abc1234] Emergency hotfix
 1 file changed, 5 insertions(+)
```

⚠️ **NOT RECOMMENDED** - Use only in emergencies! Fix formatting issues as soon as possible.

## Running Pre-commit Manually

You can run pre-commit checks without committing:

```bash
# Run on all files
$ pre-commit run --all-files

# Run on specific files
$ pre-commit run --files src/my_module/my_file.py

# Run only the ruff hook
$ pre-commit run ruff --all-files

# Run only the ruff-format hook
$ pre-commit run ruff-format --all-files
```

## Common Ruff Errors and How to Fix Them

### E501: Line too long

**Error:**
```
E501 Line too long (120 > 100)
```

**Fix:**
Break long lines into multiple lines:

```python
# Before
result = some_function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# After
result = some_function(
    arg1, arg2, arg3, arg4, arg5,
    arg6, arg7, arg8, arg9, arg10
)
```

### N801: Class name should use CapWords

**Error:**
```
N801 Class name `myClass` should use CapWords convention
```

**Fix:**
Rename class to use CamelCase:

```python
# Before
class myClass:
    pass

# After
class MyClass:
    pass
```

### E402: Module level import not at top of file

**Error:**
```
E402 Module level import not at top of file
```

**Fix:**
Move imports to the top of the file:

```python
# Before
def my_function():
    pass

import os  # ❌ Import after code

# After
import os  # ✅ Import at top

def my_function():
    pass
```

### F401: Module imported but unused

**Error:**
```
F401 [*] `os` imported but unused
```

**Fix:**
Remove the unused import:

```python
# Before
import os  # ❌ Not used anywhere
import sys

# After
import sys  # ✅ Only import what you use
```

## Disabling Pre-commit

If you need to temporarily disable pre-commit:

```bash
# Uninstall the hook (can be reinstalled later)
$ pre-commit uninstall

# Reinstall when ready
$ pre-commit install
```

## Updating Pre-commit Hooks

To update to the latest version of Ruff:

```bash
$ pre-commit autoupdate
```

This will update the version in `.pre-commit-config.yaml`.

## VS Code Integration

If you have the Ruff extension installed in VS Code, you'll see the same errors inline:

1. Errors appear with red squiggly lines
2. Hover over them to see the error message
3. Use Quick Fix (Ctrl+.) to apply auto-fixes
4. Format on save will run Ruff automatically

## Troubleshooting

### Pre-commit doesn't run when I commit

```bash
# Make sure it's installed
$ pre-commit install

# Verify it's configured
$ ls -la .git/hooks/pre-commit
```

### Pre-commit is stuck or very slow

```bash
# Clean the cache
$ pre-commit clean

# Try again
$ pre-commit run --all-files
```

### I want to ignore a specific Ruff rule

Edit `pyproject.toml` and add the rule to the ignore list:

```toml
[tool.ruff.lint]
ignore = [
    "E501",  # Ignore line length errors
]
```

## Best Practices

1. **Run pre-commit before pushing** to avoid surprises in CI/CD
2. **Fix issues immediately** rather than bypassing pre-commit
3. **Use `--all-files`** periodically to catch any issues in existing code
4. **Keep hooks updated** with `pre-commit autoupdate`
5. **Enable format-on-save** in VS Code for instant feedback

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [PEP 8 Style Guide](https://pep8.org/)
