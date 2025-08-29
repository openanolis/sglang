#!/bin/bash

# Build script for WASM plugins
# This script builds all WASM plugins in the examples/wasm_plugins directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if wasm-pack is installed
check_wasm_pack() {
    if ! command -v wasm-pack &> /dev/null; then
        print_error "wasm-pack is not installed. Please install it first:"
        echo "cargo install wasm-pack"
        exit 1
    fi
}

# Build a single WASM plugin
build_plugin() {
    local plugin_dir=$1
    local plugin_name=$(basename "$plugin_dir")
    
    print_status "Building plugin: $plugin_name"
    
    if [ ! -d "$plugin_dir" ]; then
        print_error "Plugin directory not found: $plugin_dir"
        return 1
    fi
    
    cd "$plugin_dir"
    
    # Clean previous builds
    if [ -d "target" ]; then
        rm -rf target
    fi
    
    # Build the plugin
    if wasm-pack build --target web --release; then
        print_status "Successfully built $plugin_name"
        
        # Copy the WASM file to the plugins directory
        local plugins_dir="../../plugins"
        mkdir -p "$plugins_dir"
        cp target/wasm32-unknown-unknown/release/*.wasm "$plugins_dir/${plugin_name}.wasm"
        print_status "Copied $plugin_name.wasm to $plugins_dir/"
    else
        print_error "Failed to build $plugin_name"
        return 1
    fi
    
    cd - > /dev/null
}

# Build all plugins
build_all_plugins() {
    local plugins_dir="examples/wasm_plugins"
    
    if [ ! -d "$plugins_dir" ]; then
        print_error "Plugins directory not found: $plugins_dir"
        exit 1
    fi
    
    print_status "Building all WASM plugins..."
    
    # Find all plugin directories
    for plugin_dir in "$plugins_dir"/*/; do
        if [ -d "$plugin_dir" ] && [ -f "$plugin_dir/Cargo.toml" ]; then
            build_plugin "$plugin_dir"
        fi
    done
    
    print_status "All plugins built successfully!"
}

# Test a single plugin
test_plugin() {
    local plugin_dir=$1
    local plugin_name=$(basename "$plugin_dir")
    
    print_status "Testing plugin: $plugin_name"
    
    if [ ! -d "$plugin_dir" ]; then
        print_error "Plugin directory not found: $plugin_dir"
        return 1
    fi
    
    cd "$plugin_dir"
    
    # Run tests
    if cargo test; then
        print_status "Tests passed for $plugin_name"
    else
        print_error "Tests failed for $plugin_name"
        return 1
    fi
    
    cd - > /dev/null
}

# Test all plugins
test_all_plugins() {
    local plugins_dir="examples/wasm_plugins"
    
    if [ ! -d "$plugins_dir" ]; then
        print_error "Plugins directory not found: $plugins_dir"
        exit 1
    fi
    
    print_status "Testing all WASM plugins..."
    
    # Find all plugin directories
    for plugin_dir in "$plugins_dir"/*/; do
        if [ -d "$plugin_dir" ] && [ -f "$plugin_dir/Cargo.toml" ]; then
            test_plugin "$plugin_dir"
        fi
    done
    
    print_status "All plugin tests completed!"
}

# Clean build artifacts
clean_plugins() {
    local plugins_dir="examples/wasm_plugins"
    
    print_status "Cleaning plugin build artifacts..."
    
    # Clean each plugin
    for plugin_dir in "$plugins_dir"/*/; do
        if [ -d "$plugin_dir" ] && [ -f "$plugin_dir/Cargo.toml" ]; then
            local plugin_name=$(basename "$plugin_dir")
            print_status "Cleaning $plugin_name"
            cd "$plugin_dir"
            cargo clean
            cd - > /dev/null
        fi
    done
    
    # Clean plugins directory
    if [ -d "plugins" ]; then
        rm -rf plugins
        print_status "Cleaned plugins directory"
    fi
    
    print_status "Clean completed!"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build all WASM plugins"
    echo "  test      Test all WASM plugins"
    echo "  clean     Clean build artifacts"
    echo "  all       Build and test all plugins"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 test"
    echo "  $0 clean"
    echo "  $0 all"
}

# Main script logic
main() {
    # Check if wasm-pack is installed
    check_wasm_pack
    
    case "${1:-help}" in
        build)
            build_all_plugins
            ;;
        test)
            test_all_plugins
            ;;
        clean)
            clean_plugins
            ;;
        all)
            build_all_plugins
            test_all_plugins
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
