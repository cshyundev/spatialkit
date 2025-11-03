#!/bin/bash
# Publish to production PyPI

set -e  # Exit on error

# Load environment variables
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with PYPI_TOKEN"
    echo "See .env.example for reference"
    exit 1
fi

source .env

if [ -z "$PYPI_TOKEN" ]; then
    echo "❌ Error: PYPI_TOKEN not set in .env"
    exit 1
fi

# Confirmation prompt
echo "⚠️  WARNING: You are about to publish to PRODUCTION PyPI!"
echo "Package: spatialkit"
echo "Version: $(grep '__version__' src/spatialkit/__init__.py | cut -d'"' -f2)"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Publish cancelled"
    exit 1
fi

echo ""
echo "📦 Publishing to PyPI..."

# Build first
./scripts/build.sh

# Publish to PyPI
echo ""
echo "Uploading to PyPI..."
uv publish --token "$PYPI_TOKEN"

echo ""
echo "✅ Published to PyPI!"
echo ""
echo "To install:"
echo "  pip install spatialkit"
