#!/usr/bin/env python3
"""
Script to create a .zarr.zip archive from a local zarr dataset and upload it to HuggingFace.
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

import zarr
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _validate_token():
    """Validate Hugging Face token and return API instance."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Ensure HUGGINGFACE_TOKEN is set in the environment."
        )

    hf_api = HfApi()
    try:
        user_info = hf_api.whoami(token=hf_token)
        logger.info(f"Authenticated with Hugging Face as user: {user_info['name']}")
    except Exception as e:
        raise ValueError(
            f"Failed to authenticate with Hugging Face. Check your token. Details: {e}"
        )
    return hf_api, hf_token


def _ensure_repository(hf_api, repo_id, hf_token):
    """Ensure repository exists, create if it doesn't."""
    try:
        hf_api.dataset_info(repo_id, token=hf_token)
        logger.info(f"Found existing repository: {repo_id}")
    except Exception:
        logger.info(f"Creating new dataset repository: {repo_id}")
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token)


def create_zarr_zip(
    zarr_path: Path, output_path: Path, overwrite: bool = False
) -> Path:
    """
    Create a zip archive of a Zarr directory using zarr.zip functionality.

    Args:
        zarr_path (Path): The Zarr directory to archive.
        output_path (Path): Path where the zip archive will be created.
        overwrite (bool): Whether to overwrite existing archive.

    Returns:
        Path: The path to the created archive.

    Raises:
        RuntimeError: If archive creation fails.
        ValueError: If the input folder is not a valid Zarr directory.
    """
    # Check if archive already exists
    if output_path.exists() and not overwrite:
        logger.info(f"Archive already exists: {output_path}. Skipping creation.")
        return output_path

    if output_path.exists() and overwrite:
        logger.info(f"Overwriting existing archive: {output_path}")
        output_path.unlink()  # Delete the existing archive

    try:
        # Try to open the Zarr directory to verify it's valid
        try:
            zarr.open(str(zarr_path))
        except Exception as e:
            raise ValueError(f"Not a valid Zarr directory: {zarr_path}. Error: {e}")

        # Create zip archive
        logger.info(f"Creating Zarr zip archive: {output_path}")

        # Use standard zipfile module to create zarr.zip archive
        with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the zarr directory and add all files
            for file_path in zarr_path.rglob("*"):
                if file_path.is_file():
                    # Calculate the relative path within the zarr directory
                    arcname = file_path.relative_to(zarr_path)
                    zipf.write(file_path, arcname)

        logger.info(f"Created Zarr zip archive: {output_path}")

        # Validate that the created zip can be read as a zarr store
        logger.info("Validating created zip archive can be read as zarr...")
        try:
            # Try to open the zip file as a zarr store
            zip_store = zarr.storage.ZipStore(str(output_path), mode="r")
            zarr.open(zip_store)
            logger.info("✓ Zip archive validation successful - can be read as zarr")
        except Exception as e:
            # Clean up the invalid zip file
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Created zip archive cannot be read as zarr store: {e}")

        return output_path

    except Exception as e:
        if output_path.exists():
            output_path.unlink()  # Clean up partial archive on failure
        raise RuntimeError(f"Failed to create Zarr zip archive: {e}")


def upload_zarr_to_huggingface(
    zarr_path: str,
    repo_id: str,
    target_path: str = None,
    overwrite: bool = False,
    cleanup_local: bool = True,
):
    """
    Create a zarr.zip archive and upload it to HuggingFace.

    Args:
        zarr_path (str): Path to the local zarr directory
        repo_id (str): HuggingFace repository ID (format: username/repo-name)
        target_path (str): Path in the repo where file will be uploaded. If None, uses zarr directory name
        overwrite (bool): Whether to overwrite existing files
        cleanup_local (bool): Whether to remove local archive after upload
    """
    try:
        # Convert to Path objects
        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr directory does not exist: {zarr_path}")

        # Set up output path for the zip file
        # Handle case where zarr directory already ends with .zarr
        if zarr_path.name.endswith(".zarr"):
            zip_filename = f"{zarr_path.name}.zip"
        else:
            zip_filename = f"{zarr_path.name}.zarr.zip"
        zip_path = zarr_path.parent / zip_filename

        # Set target path in repo if not provided
        if target_path is None:
            target_path = zip_filename

        # Validate token and ensure repository
        hf_api, hf_token = _validate_token()
        _ensure_repository(hf_api, repo_id, hf_token)

        # Create zarr zip archive
        logger.info(f"Creating zarr zip from: {zarr_path}")
        archive_path = create_zarr_zip(zarr_path, zip_path, overwrite=overwrite)

        # Upload to HuggingFace
        logger.info(f"Uploading {archive_path} to {repo_id}:{target_path}")

        if overwrite:
            try:
                # Delete the file if it exists and overwrite is True
                hf_api.delete_file(
                    path_in_repo=target_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token,
                )
                logger.info(f"Deleted existing file {target_path} from repository")
            except Exception as e:
                logger.debug(
                    f"File {target_path} not found in repository or couldn't be deleted: {e}"
                )

        # Upload the file
        hf_api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )

        logger.info(f"Upload completed for {archive_path} to {repo_id}:{target_path}")

        # Clean up local archive if requested
        if cleanup_local and archive_path.exists():
            logger.info(f"Removing local archive: {archive_path}")
            archive_path.unlink()

        logger.info("Upload to HuggingFace completed successfully!")

    except Exception as e:
        logger.error(f"Error uploading to HuggingFace: {e}")
        raise


def main(
    zarr_path: str = "data/Carbontracker/train/carbontracker_latlon5.625_l10_6h.zarr",
    repo_id: str = "vitusbenson/carbonbench",
    target_path: str = None,
    overwrite: bool = True,
    cleanup_local: bool = True,
):
    """
    Main function to upload zarr dataset to HuggingFace.

    Args:
        zarr_path (str): Path to the local zarr directory
        repo_id (str): HuggingFace repository ID (format: username/repo-name)
        target_path (str): Path in the repo where file will be uploaded. If None, auto-generated
        overwrite (bool): Whether to overwrite existing files
        cleanup_local (bool): Whether to remove local archive after upload
    """
    # Auto-generate target path if not provided
    if target_path is None:
        zarr_name = Path(zarr_path).name
        if "train" in zarr_path:
            target_path = f"data/train/{zarr_name}.zip"
        elif "test" in zarr_path:
            target_path = f"data/test/{zarr_name}.zip"
        elif "val" in zarr_path:
            target_path = f"data/val/{zarr_name}.zip"
        else:
            target_path = f"data/{zarr_name}.zip"

    # Upload the dataset
    upload_zarr_to_huggingface(
        zarr_path=zarr_path,
        repo_id=repo_id,
        target_path=target_path,
        overwrite=overwrite,
        cleanup_local=cleanup_local,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create and upload zarr.zip archives to HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--zarr-path",
        type=str,
        default="data/Carbontracker/train/carbontracker_latlon5.625_l10_6h.zarr",
        help="Path to the local zarr directory to upload",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="vitusbenson/carbonbench",
        help="HuggingFace repository ID (format: username/repo-name)",
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Path in the repo where file will be uploaded. If not provided, auto-generated based on zarr path",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing files in the repository",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not remove local archive after upload",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        zarr_path=args.zarr_path,
        repo_id=args.repo_id,
        target_path=args.target_path,
        overwrite=not args.no_overwrite,
        cleanup_local=not args.no_cleanup,
    )
