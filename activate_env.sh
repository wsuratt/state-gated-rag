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
