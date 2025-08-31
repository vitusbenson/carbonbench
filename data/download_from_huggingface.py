#!/usr/bin/env python3
"""
Script to download .zarr.zip archives from HuggingFace and extract them to local zarr directories.
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

import zarr
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _validate_token():
    """Validate Hugging Face token and return API instance."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning(
            "No HUGGINGFACE_TOKEN found. Attempting to download public datasets only."
        )
        return HfApi(), None

    hf_api = HfApi()
    try:
        user_info = hf_api.whoami(token=hf_token)
        logger.info(f"Authenticated with Hugging Face as user: {user_info['name']}")
    except Exception as e:
        logger.warning(
            f"Failed to authenticate with Hugging Face: {e}. Attempting public access."
        )
        return HfApi(), None
    return hf_api, hf_token


def extract_zarr_zip(zip_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    """
    Extract a zarr.zip archive to a local directory.

    Args:
        zip_path (Path): Path to the zarr.zip file
        output_dir (Path): Directory where zarr will be extracted
        overwrite (bool): Whether to overwrite existing zarr directory

    Returns:
        Path: Path to the extracted zarr directory

    Raises:
        RuntimeError: If extraction fails
        ValueError: If the zip file is not a valid zarr archive
    """
    # Determine output zarr directory name
    zarr_name = zip_path.stem  # Remove .zip extension
    if not zarr_name.endswith(".zarr"):
        zarr_name = f"{zarr_name}.zarr"

    zarr_output_path = output_dir / zarr_name

    # Check if zarr directory already exists
    if zarr_output_path.exists() and not overwrite:
        logger.info(
            f"Zarr directory already exists: {zarr_output_path}. Skipping extraction."
        )
        return zarr_output_path

    if zarr_output_path.exists() and overwrite:
        logger.info(f"Overwriting existing zarr directory: {zarr_output_path}")
        import shutil

        shutil.rmtree(zarr_output_path)

    try:
        # Create output directory
        zarr_output_path.mkdir(parents=True, exist_ok=True)

        # Extract the zip file
        logger.info(f"Extracting zarr zip archive: {zip_path} -> {zarr_output_path}")

        with zipfile.ZipFile(str(zip_path), "r") as zipf:
            zipf.extractall(str(zarr_output_path))

        # Validate that the extracted directory is a valid zarr
        logger.info("Validating extracted zarr directory...")
        try:
            zarr.open(str(zarr_output_path))
            logger.info("✓ Extracted zarr directory validation successful")
        except Exception as e:
            # Clean up invalid extraction
            import shutil

            shutil.rmtree(zarr_output_path)
            raise ValueError(f"Extracted directory is not a valid zarr store: {e}")

        logger.info(f"Successfully extracted zarr to: {zarr_output_path}")
        return zarr_output_path

    except Exception as e:
        # Clean up partial extraction on failure
        if zarr_output_path.exists():
            import shutil

            shutil.rmtree(zarr_output_path)
        raise RuntimeError(f"Failed to extract zarr zip archive: {e}")


def download_and_extract_zarr(
    repo_id: str,
    filename: str,
    local_dir: str = "./data",
    overwrite: bool = False,
    keep_zip: bool = False,
):
    """
    Download a zarr.zip file from HuggingFace and extract it.

    Args:
        repo_id (str): HuggingFace repository ID (format: username/repo-name)
        filename (str): Path to the file in the repository (e.g., "data/train/dataset.zarr.zip")
        local_dir (str): Local directory where files will be downloaded and extracted
        overwrite (bool): Whether to overwrite existing files/directories
        keep_zip (bool): Whether to keep the downloaded zip file after extraction
    """
    try:
        # Validate token (optional for public repos)
        hf_api, hf_token = _validate_token()

        # Create local directory
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download the file
        logger.info(f"Downloading {filename} from {repo_id}...")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=hf_token,
            force_download=overwrite,
        )

        downloaded_path = Path(downloaded_path)
        logger.info(f"Downloaded to: {downloaded_path}")

        # Extract the zarr zip
        logger.info("Extracting zarr archive...")
        zarr_dir = extract_zarr_zip(
            zip_path=downloaded_path, output_dir=local_dir, overwrite=overwrite
        )

        # Clean up zip file if not keeping it
        if not keep_zip:
            logger.info(f"Removing downloaded zip file: {downloaded_path}")
            downloaded_path.unlink()

        logger.info(f"Download and extraction completed! Zarr available at: {zarr_dir}")
        return zarr_dir

    except Exception as e:
        logger.error(f"Error downloading and extracting zarr: {e}")
        raise


def download_and_extract_all_zarr(
    repo_id: str,
    local_dir: str = "./data",
    overwrite: bool = False,
    keep_zip: bool = False,
):
    """
    Download the entire repository and extract all zarr.zip files found.

    Args:
        repo_id (str): HuggingFace repository ID (format: username/repo-name)
        local_dir (str): Local directory where repo will be downloaded and zarr files extracted
        overwrite (bool): Whether to overwrite existing files/directories
        keep_zip (bool): Whether to keep the downloaded zip files after extraction
    """
    try:
        # Validate token (optional for public repos)
        hf_api, hf_token = _validate_token()

        # Create local directory
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download the entire repository
        logger.info(f"Downloading entire repository {repo_id}...")

        repo_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=hf_token,
            force_download=overwrite,
        )

        repo_path = Path(repo_path)
        logger.info(f"Repository downloaded to: {repo_path}")

        # Find all zarr.zip files in the downloaded repository
        zarr_zip_files = list(repo_path.rglob("*.zarr.zip"))

        if not zarr_zip_files:
            logger.warning("No zarr.zip files found in the repository")
            return []

        logger.info(f"Found {len(zarr_zip_files)} zarr.zip files to extract")

        extracted_zarr_dirs = []

        # Extract each zarr.zip file
        for zip_file in zarr_zip_files:
            logger.info(f"Processing: {zip_file}")

            try:
                # Extract to the same directory as the zip file
                zarr_dir = extract_zarr_zip(
                    zip_path=zip_file, output_dir=zip_file.parent, overwrite=overwrite
                )
                extracted_zarr_dirs.append(zarr_dir)

                # Clean up zip file if not keeping it
                if not keep_zip:
                    logger.info(f"Removing zip file: {zip_file}")
                    zip_file.unlink()

            except Exception as e:
                logger.error(f"Failed to extract {zip_file}: {e}")
                continue

        logger.info(
            f"Repository download and extraction completed! Extracted {len(extracted_zarr_dirs)} zarr datasets"
        )
        for zarr_dir in extracted_zarr_dirs:
            logger.info(f"  - {zarr_dir}")

        return extracted_zarr_dirs

    except Exception as e:
        logger.error(f"Error downloading and extracting repository: {e}")
        raise


def main(
    repo_id: str = "vitusbenson/carbonbench",
    filename: str = "data/train/carbontracker_latlon5.625_l10_6h.zarr.zip",
    local_dir: str = "./data",
    overwrite: bool = False,
    keep_zip: bool = False,
    download_all: bool = False,
):
    """
    Main function to download and extract zarr dataset(s) from HuggingFace.

    Args:
        repo_id (str): HuggingFace repository ID
        filename (str): Path to the zarr.zip file in the repository (ignored if download_all=True)
        local_dir (str): Local directory for download and extraction
        overwrite (bool): Whether to overwrite existing files/directories
        keep_zip (bool): Whether to keep the downloaded zip file
        download_all (bool): Whether to download entire repo and extract all zarr.zip files
    """
    if download_all:
        download_and_extract_all_zarr(
            repo_id=repo_id,
            local_dir=local_dir,
            overwrite=overwrite,
            keep_zip=keep_zip,
        )
    else:
        download_and_extract_zarr(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            overwrite=overwrite,
            keep_zip=keep_zip,
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract zarr.zip archives from HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="vitusbenson/carbonbench",
        help="HuggingFace repository ID (format: username/repo-name)",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="data/train/carbontracker_latlon5.625_l10_6h.zarr.zip",
        help="Path to the zarr.zip file in the repository",
    )

    parser.add_argument(
        "--local-dir",
        type=str,
        default="./data",
        help="Local directory where files will be downloaded and extracted",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files and directories",
    )

    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )

    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download entire repository and extract all zarr.zip files found",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.local_dir,
        overwrite=args.overwrite,
        keep_zip=args.keep_zip,
        download_all=args.download_all,
    )