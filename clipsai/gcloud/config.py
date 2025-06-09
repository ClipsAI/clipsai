# clipsai/gcloud/config.py
import os
import logging

from clipsai.utils.exceptions import ConfigError

# Environment variable names
ENV_VAR_GCLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
ENV_VAR_GCLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION" # e.g., "us-central1" for some services
ENV_VAR_GCLOUD_TEMP_GCS_BUCKET = "GOOGLE_CLOUD_TEMP_GCS_BUCKET_NAME"


class GCloudConfig:
    """
    Configuration for Google Cloud services used by ClipsAI.

    Reads configuration from environment variables or direct parameters.
    """
    def __init__(
        self,
        project_id: str = None,
        location: str = None,
        temp_gcs_bucket_name: str = None,
    ):
        """
        Initializes Google Cloud configuration.

        Parameters
        ----------
        project_id : str, optional
            Google Cloud Project ID. If None, tries to read from
            GOOGLE_CLOUD_PROJECT environment variable.
        location : str, optional
            Google Cloud location/region (e.g., "us-central1"). If None, tries to
            read from GOOGLE_CLOUD_LOCATION environment variable. Some services
            are global and may not require this.
        temp_gcs_bucket_name : str, optional
            Name of the GCS bucket to use for temporary file uploads (e.g., for
            Video Intelligence API processing of large local files). If None,
            tries to read from GOOGLE_CLOUD_TEMP_GCS_BUCKET_NAME environment variable.
        """
        self.project_id = project_id or os.environ.get(ENV_VAR_GCLOUD_PROJECT)
        self.location = location or os.environ.get(ENV_VAR_GCLOUD_LOCATION)
        self.temp_gcs_bucket_name = temp_gcs_bucket_name or                                     os.environ.get(ENV_VAR_GCLOUD_TEMP_GCS_BUCKET)

        if not self.project_id:
            # Many Google Cloud services require a project ID.
            # Making this a warning for now, but some services will fail without it.
            # Specific services should raise errors if it's missing and required.
            logging.warning(
                f"Google Cloud Project ID not set. Please set the "
                f"{ENV_VAR_GCLOUD_PROJECT} environment variable or pass it explicitly "
                f"if required by the Google Cloud service being used."
            )
        else:
            logging.info(f"Using Google Cloud Project: {self.project_id}")

        if self.location:
            logging.info(f"Using Google Cloud Location: {self.location}")
        else:
            # Some services are regional, some are global.
            logging.info(
                "Google Cloud Location not explicitly set. Using client library "
                "defaults or assuming global service endpoint if applicable."
            )

        # temp_gcs_bucket_name is optional for now; only components needing GCS
        # upload will check for it more strictly.
        if self.temp_gcs_bucket_name:
            logging.info(f"Using GCS bucket for temporary files: {self.temp_gcs_bucket_name}")
