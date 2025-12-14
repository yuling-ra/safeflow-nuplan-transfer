#!/usr/bin/env bash
# Run trajectory reconstruction visualization with best configurations

set -e

# Default parameters
DATA="../Compress_Test/test_complete.npz"
SAMPLE_IDX=0
OUT="./reconstruction_comparison_$(date +%Y%m%d_%H%M%S)"

# Help message
show_help() {
    cat << EOF
Usage: $0 [options]

Visualize trajectory reconstruction comparison using best configurations

Options:
    -d, --data PATH         Data file path (default: ../Compress_Test/test_complete.npz)
    -s, --sample IDX        Sample index to visualize (default: 0)
    -o, --out DIR           Output directory (default: ./reconstruction_comparison_TIMESTAMP)
    -h, --help              Show this help message

Best Configurations:
    Strategy 1 (Fixed q High-Fidelity):
      - M: q=128, dq=64, tau=64
      - Degree: q=5, dq=3, tau=5
      - Lambda: q=1e-6, dq=1e-6, tau=1e-5
      - Latent: 1792 dims
      - WRMSE: 0.150360

    Strategy 2 (1k Budget):
      - M: q=110, dq=16, tau=16
      - Degree: q=5, dq=3, tau=5
      - Lambda: q=1e-6, dq=1e-6, tau=1e-5
      - Latent: 994 dims
      - WRMSE: 0.269323

Example:
    # Default: sample 0
    $0

    # Visualize sample 10
    $0 --sample 10

    # Custom data file and output
    $0 --data /path/to/data.npz --out ./my_results --sample 5

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA="$2"
            shift 2
            ;;
        -s|--sample)
            SAMPLE_IDX="$2"
            shift 2
            ;;
        -o|--out)
            OUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check data file
if [ ! -f "$DATA" ]; then
    echo "Error: Data file not found: $DATA"
    exit 1
fi

echo "========================================================================"
echo "Trajectory Reconstruction Visualization"
echo "========================================================================"
echo "Data file: $DATA"
echo "Sample index: $SAMPLE_IDX"
echo "Output directory: $OUT"
echo "========================================================================"
echo ""

# Run visualization with best configurations
python3 visualize_reconstruction.py \
    --data "$DATA" \
    --sample-idx "$SAMPLE_IDX" \
    --out "$OUT" \
    --strategy1-Mq 128 \
    --strategy1-Mdq 64 \
    --strategy1-Mtau 64 \
    --strategy1-deg-q 5 \
    --strategy1-deg-dq 3 \
    --strategy1-deg-tau 5 \
    --strategy1-lam-q 1e-6 \
    --strategy1-lam-dq 1e-6 \
    --strategy1-lam-tau 1e-5 \
    --strategy2-Mq 110 \
    --strategy2-Mdq 16 \
    --strategy2-Mtau 16 \
    --strategy2-deg-q 5 \
    --strategy2-deg-dq 3 \
    --strategy2-deg-tau 5 \
    --strategy2-lam-q 1e-6 \
    --strategy2-lam-dq 1e-6 \
    --strategy2-lam-tau 1e-5

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Visualization Complete!"
    echo "========================================================================"
    echo "Results saved to: $OUT"
    echo ""
    echo "Generated plots:"
    ls -lh "$OUT"/*.png | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
    echo "Open the plots to compare original vs reconstructed trajectories!"
    echo "========================================================================"
else
    echo ""
    echo "Error: Visualization failed"
    exit 1
fi
