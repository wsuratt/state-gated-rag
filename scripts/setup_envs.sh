#!/bin/bash
# Setup script for WebShop and ALFWorld environments

set -e

echo "=== Setting up SSM-Agent environments ==="

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "ssm-agent"; then
    echo "Creating conda environment..."
    conda create -n ssm-agent python=3.11 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ssm-agent

# Install base requirements
echo "Installing base requirements..."
pip install -r requirements.txt

# Install additional dependencies
echo "Installing additional dependencies..."
pip install gdown

# Install spacy model (required for WebShop)
echo "Installing spacy model..."
python -m spacy download en_core_web_sm || true

# Setup Java (required for WebShop search indexing)
# Java 21 is required for pyserini/Anserini compatibility
echo "Checking Java installation..."
if [ -d "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
    export JVM_PATH=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Java 21 found: $(java -version 2>&1 | head -1)"
elif [ -d "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
    export JVM_PATH=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Java 17 found: $(java -version 2>&1 | head -1)"
    echo "WARNING: Java 21 is recommended. Install with: brew install openjdk@21"
else
    echo "WARNING: Java not found. WebShop search indexing will not work."
    echo "Install Java 21 with: brew install openjdk@21"
fi

# Clone WebShop repo for data and environment
echo "Setting up WebShop..."
mkdir -p external
cd external

if [ ! -d "WebShop" ]; then
    git clone https://github.com/princeton-nlp/WebShop.git
fi

cd WebShop

# Download data files if not present
if [ ! -f "data/items_shuffle_1000.json" ]; then
    echo "Downloading WebShop data files..."
    mkdir -p data

    # Download small dataset files using gdown
    gdown --id 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O data/items_shuffle_1000.json
    gdown --id 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O data/items_ins_v2_1000.json
    gdown --id 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O -O data/items_human_ins.json

    echo "WebShop data downloaded."
else
    echo "WebShop data already exists."
fi

# Build search index if not present
if [ ! -d "search_engine/indexes_1k" ] || [ -z "$(ls -A search_engine/indexes_1k 2>/dev/null)" ]; then
    echo "Building WebShop search index..."
    cd search_engine

    # Generate resource files from product data
    python convert_product_file_format.py

    # Build 1k Lucene index
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input resources_1k \
        --index indexes_1k \
        --generator DefaultLuceneDocumentGenerator \
        --threads 1 \
        --storePositions --storeDocvectors --storeRaw

    cd ..
    echo "WebShop search index built."
else
    echo "WebShop search index already exists."
fi

cd ../..

# Install ALFWorld
echo "Setting up ALFWorld..."
cd external

if [ ! -d "alfworld" ]; then
    git clone https://github.com/alfworld/alfworld.git
fi

cd alfworld

# Check if alfworld is already installed
if ! pip show alfworld > /dev/null 2>&1; then
    echo "Installing ALFWorld..."
    pip install -e .
    # Download game files
    alfworld-download
else
    echo "ALFWorld already installed."
fi

cd ../..

# Create activation script with proper environment variables
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activate SSM-Agent environment with proper paths

# Activate conda
eval "$(conda shell.bash hook)"
conda activate ssm-agent

# Set Java 21 (required for pyserini/Anserini compatibility)
# Must use the actual JDK contents path inside libexec
if [ -d "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
    export JVM_PATH=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib
elif [ -d "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
    export JVM_PATH=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib
fi

if [ -n "$JAVA_HOME" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

# Add WebShop to Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/external/WebShop:$PYTHONPATH"

echo "Environment activated!"
echo "  JAVA_HOME: $JAVA_HOME"
echo "  JVM_PATH: $JVM_PATH"
echo "  PYTHONPATH includes: ${SCRIPT_DIR}/external/WebShop"
EOF

chmod +x activate_env.sh

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source activate_env.sh"
echo ""
echo "Or manually:"
echo "  conda activate ssm-agent"
echo "  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
echo "  export JVM_PATH=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib"
echo "  export PYTHONPATH=\"\$(pwd)/external/WebShop:\$PYTHONPATH\""
