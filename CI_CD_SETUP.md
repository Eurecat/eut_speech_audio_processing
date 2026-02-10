# CI/CD Pipeline Setup Guide

This guide explains how to reuse and adapt the CI/CD pipeline from this repository for your own ROS 2 projects.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [GitHub Secrets Configuration](#github-secrets-configuration)
4. [Workflow Components](#workflow-components)
5. [Adapting for Your Project](#adapting-for-your-project)
6. [Badge Setup](#badge-setup)
7. [Local Verification](#local-verification)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The CI/CD pipeline provides:
- ✅ **Automated Docker image building** on every push/PR
- 🧪 **Comprehensive testing** (C++ GoogleTest + Python pytest)
- 📊 **Code coverage tracking** with HTML reports and badges
- 📦 **Docker Hub deployment** from protected branches
- 🏷️ **Dynamic badges** for build status, tests, and coverage

**Pipeline Flow:**
```
Push/PR → Build Docker → Build Packages → Run Tests → Generate Coverage → Create Badges → Push to Docker Hub
```

---

## Prerequisites

Before setting up the CI/CD pipeline, ensure you have:

### Repository Requirements
- A GitHub repository with ROS 2 packages
- A Dockerfile for your development environment
- Package tests (C++ with GoogleTest, Python with pytest)

### Docker Hub Account (Optional)
- Docker Hub account for image hosting
- Docker Hub organization or personal namespace

### GitHub Repository Access
- Admin access to repository settings
- Ability to create/manage GitHub Actions secrets

---

## GitHub Secrets Configuration

Navigate to your repository: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

### Required Secrets

#### 1. Docker Hub Authentication

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username | `myusername` |
| `DOCKERHUB_TOKEN` | Docker Hub access token ([create here](https://hub.docker.com/settings/security)) | `dckr_pat_xxxxx...` |
| `DOCKERHUB_ORG` | Docker Hub org or username for image push | `myorganization` |

**Creating Docker Hub Access Token:**
1. Go to https://hub.docker.com/settings/security
2. Click "New Access Token"
3. Name it (e.g., "GitHub Actions")
4. Select permissions: **Read & Write**
5. Copy the token (shown only once!)
6. Add to GitHub secrets as `DOCKERHUB_TOKEN`

#### 2. Optional Secrets

| Secret Name | Purpose | When Needed |
|-------------|---------|-------------|
| `PRIVATE_ACTIONS_DEPLOY` | SSH key for private repos | If cloning private dependencies during build |

### Verifying Secrets

After adding secrets, verify they appear in:
**Settings** → **Secrets and variables** → **Actions** → **Repository secrets**

---

## Workflow Components

The workflow file is located at [`.github/workflows/docker-build.yml`](.github/workflows/docker-build.yml).

### Key Stages

#### Stage 1: Docker Image Build
```yaml
- name: Build Docker image
  run: cd ./Docker && ./build_container.sh
```
- Builds your Docker image using your Dockerfile
- Verifies the image was created successfully

#### Stage 2: Package Build & Test
```yaml
- name: Build & test workspace
  run: |
    colcon build --symlink-install \
      --packages-select audio_stream_manager \
      --cmake-args -DCMAKE_CXX_FLAGS='--coverage' ...
    colcon test --event-handlers console_direct+ ...
```
- Builds C++ packages with coverage flags
- Builds Python packages
- Runs all tests with verbose output

#### Stage 3: Coverage Generation
```yaml
- name: Generate coverage reports
  run: docker run --rm ... bash -c "/ci_cd_coverage.sh"
```
- Generates HTML coverage reports
- Creates LCOV format for badge generation
- Converts to Cobertura XML for PR comments

#### Stage 4: Badge Creation
```yaml
- name: Create test badge
  run: # Parses test results and creates badge JSON
- name: Create coverage badge
  run: # Extracts coverage percentage and creates badge JSON
```
- Parses test results for pass/fail counts
- Extracts coverage percentages from LCOV files
- Generates shields.io-compatible JSON files

#### Stage 5: Badge Publishing
```yaml
- name: Publish badges to badges branch
  run: # Commits badge JSONs to 'badges' branch
```
- Creates orphan `badges` branch
- Stores badge JSON files per source branch
- Enables shields.io endpoint badges in README

#### Stage 6: Docker Hub Deployment
```yaml
- name: Push to Docker Hub
  if: github.ref == 'refs/heads/jazzy-devel'
  run: docker push ...
```
- Tags image with org/repo:tag format
- Pushes to Docker Hub on jazzy-devel branch only

**Please note**: Same tag does not entail you are deleting old images in dockerhub pushed before: you should take care of deleting them in the panel from time to time or upgrade the github action workflow ;-)

---

## Adapting for Your Project

### Step 1: Copy Workflow File

```bash
# Copy the workflow file to your repo
cp .github/workflows/docker-build.yml YOUR_REPO/.github/workflows/
```

### Step 2: Update Branch Names

Edit the workflow to match your branch strategy:

```yaml
on:
  push:
    branches:
      - develop        # Your dev branch
  pull_request:
    branches:
      - jazzy-devel
```

### Step 3: Customize Package Names

**Package names are now centralized!** Edit the environment variables at the top of the workflow file:

```yaml
env:
  # Package Configuration - Define your ROS2 packages here
  CPP_PACKAGES: "YOUR_CPP_PACKAGE"
  PY_PACKAGES: "YOUR_PY_PACKAGE"
```

**For multiple packages** (space-separated):
```yaml
env:
  CPP_PACKAGES: "pkg1 pkg2 pkg3"
  PY_PACKAGES: "py_pkg1 py_pkg2"
```

The workflow automatically uses these variables throughout all steps (build, test, coverage). No need to update package names in multiple places!

**Similarly, update [`Docker/ci_cd_coverage.sh`](Docker/ci_cd_coverage.sh):**
```bash
# PACKAGE CONFIGURATION
CPP_PACKAGES=("YOUR_CPP_PACKAGE")
PY_PACKAGES=("YOUR_PY_PACKAGE")
```

### Step 4: Adjust Docker Image Names

Update Docker image verification:

```yaml
- name: Verify image was built
  run: |
    if docker images | grep -q "YOUR_IMAGE_NAME"; then
      echo "✅ Image YOUR_IMAGE_NAME successfully built"
```

### Step 5: Update Coverage Script Path

If your coverage script has a different name or location:

```yaml
- name: Generate coverage reports
  run: |
    docker run --rm ... bash -c "/path/to/your_coverage_script.sh"
```

### Step 6: Customize Badge Branches

Update badge publishing conditions to match your branches:

```yaml
- name: Publish badges to badges branch
  if: always() && (github.ref == 'refs/heads/jazzy-devel' || github.ref == 'refs/heads/develop')
  run: |
    if [ "${{ github.ref }}" == "refs/heads/jazzy-devel" ]; then
      BRANCH_NAME="jazzy-devel"
    elif [ "${{ github.ref }}" == "refs/heads/develop" ]; then
      BRANCH_NAME="develop"
    fi
```

### Step 7: Configure Docker Hub Push

Control which branches can push to Docker Hub:

```yaml
- name: Push to Docker Hub
  if: github.ref == 'refs/heads/jazzy-devel'  # Only from jazzy-devel
  run: docker push ${DOCKERHUB_ORG}/your-image:${TAG}
```

**Options:**
- `refs/heads/jazzy-devel` - Only jazzy-devel branch
- `refs/heads/*` - All branches
- `github.ref == 'refs/heads/jazzy-devel' || github.ref == 'refs/heads/develop'` - Multiple branches

---

## Badge Setup

### Step 1: Create Badges Branch

The workflow automatically creates an orphan `badges` branch on first run. No manual action needed!

### Step 2: Add Badges to README

Add to your `README.md`:

```markdown
# Your Project Name

[![Build Status](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/docker-build.yml/badge.svg?branch=jazzy-devel)](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/docker-build.yml)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/badges/jazzy-devel/test-badge.json)](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/docker-build.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/badges/jazzy-devel/coverage-badge.json)](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/docker-build.yml)
```

**Replace:**
- `YOUR_ORG` → Your GitHub organization/username
- `YOUR_REPO` → Your repository name
- `jazzy-devel` → Your branch name (if different)

### Badge Variants

**For multiple branches:**
```markdown
## Main Branch
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/badges/jazzy-devel/test-badge.json)](...)

## Develop Branch
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/badges/develop/test-badge.json)](...)
```

**Build status only (no custom badges):**
```markdown
[![Build](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/docker-build.yml/badge.svg)](...)
```

---

## Local Verification

### Quick Test Script

Copy the verification script to your container:

```dockerfile
# In your Dockerfile
COPY Docker/quick_test_coverage.sh /quick_test_coverage.sh
RUN chmod +x /quick_test_coverage.sh
```

### Usage

```bash
# Inside your container
/quick_test_coverage.sh --cpp YOUR_CPP_PKG --python YOUR_PY_PKG

# Test all packages
/quick_test_coverage.sh --all

# Clean build before testing
/quick_test_coverage.sh --all --clean
```

### Script Benefits
- ✅ Mirrors GitHub Actions workflow exactly
- 🧪 Runs full build → test → coverage cycle
- 📊 Generates coverage reports locally
- ⚡ Fast feedback before pushing

---

## Troubleshooting

### Common Issues

#### 1. **Badges Show "invalid"**

**Cause:** Badge JSON files not generated or wrong branch name

**Solution:**
1. Check workflow ran successfully on your branch
2. Verify badges branch exists: `git ls-remote origin badges`
3. Check badge URL matches branch name in workflow
4. Wait ~5 min for GitHub CDN to update

#### 2. **Docker Hub Push Fails**

**Cause:** Missing or incorrect secrets

**Solution:**
```bash
# Verify secrets are set
# Settings → Secrets → Actions → Repository secrets
- DOCKERHUB_USERNAME ✓
- DOCKERHUB_TOKEN ✓  
- DOCKERHUB_ORG ✓

# Test locally
docker login -u $DOCKERHUB_USERNAME
docker push ${DOCKERHUB_ORG}/image:tag
```

#### 3. **Coverage Badge Shows 0%**

**Cause:** Coverage files not generated in expected location

**Solution:**
1. Check "Generate coverage reports" step logs
2. Verify `/ci_cd_coverage.sh` exists in container
3. Check Python package uses pytest-cov:
   ```toml
   [tool.pytest.ini_options]
   addopts = "--cov=YOUR_PACKAGE --cov-report=lcov"
   ```
4. Verify C++ built with coverage flags

#### 4. **Tests Pass Locally but Fail in CI**

**Cause:** Environment differences

**Solution:**
1. Run tests in Docker container locally:
   ```bash
   docker run --rm -v $(pwd):/workspace -w /workspace YOUR_IMAGE \
     bash -c "colcon test --packages-select YOUR_PKG"
   ```
2. Check for hardcoded paths
3. Verify all dependencies in Dockerfile
4. Check ROS environment sourced correctly

#### 5. **Workflow Triggers on Wrong Branches**

**Cause:** Branch filter mismatch

**Solution:**
Update `on.push.branches` and `on.pull_request.branches`:
```yaml
on:
  push:
    branches:
      - jazzy-devel
      - your-branch-name
```

### Debug Mode

Enable workflow debugging:
1. Repository **Settings** → **Secrets** → **Actions**
2. Add repository secret: `ACTIONS_STEP_DEBUG` = `true`
3. Re-run workflow for detailed logs

### Getting Help

If issues persist:
1. Check [GitHub Actions logs](../../actions)
2. Review [workflow file](.github/workflows/docker-build.yml)
3. Compare with [this reference implementation](.)
4. Open an issue with:
   - Workflow run link
   - Error messages
   - Your workflow customizations

---

## Best Practices

### Security
- ✅ Never commit secrets to code
- ✅ Use GitHub Secrets for sensitive data
- ✅ Limit Docker Hub token to Read & Write only
- ✅ Rotate tokens periodically

### Performance
- ⚡ Use `--symlink-install` for faster builds
- ⚡ Cache Docker layers efficiently
- ⚡ Run only affected package tests when possible

### Maintainability
- 📝 Document custom workflow changes
- 🏷️ Use semantic versioning for Docker tags
- 🔄 Keep workflow in sync with local scripts
- ✅ Test workflow changes in feature branches first

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Access Tokens](https://docs.docker.com/docker-hub/access-tokens/)
- [Shields.io Endpoint Badges](https://shields.io/endpoint)
- [ROS 2 Testing Guide](https://docs.ros.org/en/rolling/Tutorials/Intermediate/Testing/Testing-Main.html)

---

## Quick Reference

### Files to Copy
```
.github/workflows/docker-build.yml    # Main workflow
Docker/quick_test_coverage.sh         # Local verification script
Docker/ci_cd_coverage.sh               # Coverage generation script
```

### Secrets to Set
```
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
DOCKERHUB_ORG
```

### Workflow Customization Points
1. **Package configuration** (`env.CPP_PACKAGES`, `env.PY_PACKAGES`) - **Start here!**
2. Branch names (`on.push.branches`)
3. Docker image names
4. Badge branch names
5. Docker Hub push conditions
6. Coverage script packages ([`Docker/ci_cd_coverage.sh`](Docker/ci_cd_coverage.sh))

### Badge URLs Format
```
Build:    https://github.com/ORG/REPO/actions/workflows/WORKFLOW.yml/badge.svg?branch=BRANCH
Tests:    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ORG/REPO/badges/BRANCH/test-badge.json
Coverage: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ORG/REPO/badges/BRANCH/coverage-badge.json
```

---

**Questions or issues?** Feel free to open an issue or refer to the jazzy-devel [README](README.md) for testing details.
