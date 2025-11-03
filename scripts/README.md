# Build and Publish Scripts

Scripts for building and publishing spatialkit to PyPI.

## Setup

1. Create a `.env` file in the project root (see `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. Add your PyPI tokens to `.env`:
   - Get TestPyPI token: https://test.pypi.org/manage/account/token/
   - Get PyPI token: https://pypi.org/manage/account/token/

## Usage

### Build Package

```bash
./scripts/build.sh
```

This will:
- Clean previous builds
- Build wheel and source distribution
- Output files to `dist/`

### Publish to TestPyPI

```bash
./scripts/publish_test.sh
```

Test the installation:
```bash
pip install --index-url https://test.pypi.org/simple/ spatialkit
```

### Publish to Production PyPI

```bash
./scripts/publish.sh
```

**Warning**: This publishes to production PyPI. Make sure:
- Version is correct in `src/spatialkit/__init__.py`
- CHANGELOG.md is updated
- All tests pass
- Code is committed and tagged

## Manual Publishing (Alternative)

If you prefer manual control:

```bash
# Build
uv build

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/ --token $TESTPYPI_TOKEN

# Publish to PyPI
uv publish --token $PYPI_TOKEN
```
