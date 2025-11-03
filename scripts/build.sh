#!/bin/bash
# Build script for spatialkit package

set -e  # Exit on error

echo "🔨 Building spatialkit..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build the package
echo "Building wheel and sdist..."
uv build

echo "✅ Build complete!"
echo ""
echo "Generated files:"
ls -lh dist/
