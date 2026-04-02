#!/bin/bash
# GeoAI Digital Asset Pipeline - Setup Script
# ============================================
# This script sets up the development environment for the GeoAI pipeline.

set -e

echo "=============================================="
echo "GeoAI Digital Asset Pipeline - Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -n "Checking Python version... "
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 10 ]); then
    echo -e "${RED}Failed${NC}"
    echo "Error: Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo -e "${GREEN}OK${NC} ($PYTHON_VERSION)"

# Check for GDAL
echo -n "Checking GDAL... "
if command -v gdalinfo &> /dev/null; then
    GDAL_VERSION=$(gdalinfo --version | head -1 | cut -d' ' -f2 | tr -d ',')
    echo -e "${GREEN}OK${NC} ($GDAL_VERSION)"
else
    echo -e "${YELLOW}Warning${NC} (GDAL not found, some features may not work)"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install base dependencies
echo ""
echo "Installing base dependencies..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}Base dependencies installed${NC}"

# Install package in development mode
echo ""
echo "Installing package in development mode..."
pip install -e . --quiet
echo -e "${GREEN}Package installed${NC}"

# Install development dependencies
echo ""
read -p "Install development dependencies? (pytest, black, flake8, etc.) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install pytest pytest-cov pytest-mock black flake8 mypy isort --quiet
    echo -e "${GREEN}Development dependencies installed${NC}"
fi

# Create directories
echo ""
echo "Creating directory structure..."
mkdir -p data/raw data/processed outputs models logs notebooks
touch data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep models/.gitkeep logs/.gitkeep
echo -e "${GREEN}Directories created${NC}"

# Copy sample config if not exists
if [ ! -f "config.local.yaml" ]; then
    cp config.yaml config.local.yaml 2>/dev/null || true
    echo -e "${GREEN}Sample configuration created${NC}"
fi

# Run tests
echo ""
read -p "Run tests to verify installation? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    pytest tests/ -v --tb=short || echo -e "${YELLOW}Some tests failed${NC}"
fi

# Print summary
echo ""
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the demo:"
echo "     python -m src.main demo"
echo ""
echo "  3. Or use the CLI:"
echo "     geoai --help"
echo ""
echo "  4. Check the documentation:"
echo "     cat README.md"
echo ""
echo "=============================================="
