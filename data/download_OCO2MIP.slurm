#!/bin/bash
#SBATCH --partition=work
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --job-name=OCO2MIP_download
#SBATCH --mail-type=END,FAIL
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data/OCO2MIP_download_%A.out

# ================================
# OCO2MIP Input Data Download Script
# ================================
# Downloads:
#   1. OCO-2 10-sec data (v11.2)
#   2. OCO-3 10-sec data (v11)
#   3. TCCON time-averaged data (Caltech)
#   4. ObsPack CO2 product (requires access link)
#
# Usage: sbatch download_OCO2MIP.slurm
# ================================

set -euo pipefail
echo "Starting OCO2MIP download job at $(date)"

# -------- Configuration --------
BASE_DIR=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP
mkdir -p "${BASE_DIR}"/{OCO2,OCO3,TCCON,ObsPack}

download_file() {
    local url=$1
    local outdir=$2
    local name=$(basename "$url")
    echo "Downloading ${name}..."
    wget --progress=dot:giga -P "${outdir}" "${url}" 2>&1 | grep --line-buffered -E "^(--|Length|Saving to|HTTP|Proxy|Connecting|Resolving)" || true
}

# -------- OCO-2 --------
download_file "https://gml.noaa.gov/aftp/user/andy/OCO-2/OCO2_b11.2_10sec_GOOD_r2.nc4" "${BASE_DIR}/OCO2"

# -------- OCO-3 --------
download_file "https://gml.noaa.gov/aftp/user/andy/OCO-2/OCO3_b11_10sec_GOOD_r2.nc4" "${BASE_DIR}/OCO3"

# -------- TCCON time-averaged data --------
download_file "https://data.caltech.edu/records/zr28z-s4y31/files/tccon_timeaverages_R20250609.tgz" "${BASE_DIR}/TCCON"

# Extract if needed
TCCON_ARCHIVE="${BASE_DIR}/TCCON/tccon_timeaverages_R20250609.tgz"
if [[ -f "$TCCON_ARCHIVE" ]]; then
    echo "Extracting TCCON archive..."
    tar -xzf "$TCCON_ARCHIVE" -C "${BASE_DIR}/TCCON"
else
    echo "⚠️  TCCON archive not found — skipping extraction."
fi

# -------- ObsPack CO2 (requires manual access) --------
OBSPACK_URL="<YOUR_PRIVATE_OBSPACK_URL>"

if [[ "${OBSPACK_URL}" != "<YOUR_PRIVATE_OBSPACK_URL>" ]]; then
    download_file "${OBSPACK_URL}" "${BASE_DIR}/ObsPack"
    echo "Extracting ObsPack archive..."
    tar -xzf "${BASE_DIR}/ObsPack/"*.tar.gz -C "${BASE_DIR}/ObsPack"
else
    echo "⚠️  Skipping ObsPack download (no access URL provided)."
    echo "    Please fill in your private ObsPack link in this script once NOAA grants access."
fi

# -------- Summary --------
echo ""
echo "================== DOWNLOAD SUMMARY =================="
find "${BASE_DIR}" -maxdepth 2 -type f -exec ls -lh {} \; | awk '{printf "%-60s %8s\n", $9, $5}'
echo "======================================================"
echo "All downloads complete at $(date)"
