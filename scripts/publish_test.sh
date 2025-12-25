#!/bin/bash
# Publish to TestPyPI

set -e  # Exit on error

# Load environment variables
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with TESTPYPI_TOKEN"
    echo "See .env.example for reference"
    exit 1
fi

source .env

if [ -z "$TESTPYPI_TOKEN" ]; then
    echo "❌ Error: TESTPYPI_TOKEN not set in .env"
    exit 1
fi

# Confirmation prompt
echo "📦 You are about to publish to TestPyPI"
echo "Package: spatialkit"
echo "Version: $(grep '__version__' src/spatialkit/__init__.py | cut -d'"' -f2)"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Publish cancelled"
    exit 1
fi

echo ""
echo "📦 Publishing to TestPyPI..."

# Build first
./scripts/build.sh

# Publish to TestPyPI
echo ""
echo "Uploading to TestPyPI..."
uv publish \
    --publish-url https://test.pypi.org/legacy/ \
    --token "$TESTPYPI_TOKEN"

echo ""
echo "✅ Published to TestPyPI!"
echo ""
echo "To install from TestPyPI:"
echo "  pip install --index-url https://test.pypi.org/simple/ spatialkit"
