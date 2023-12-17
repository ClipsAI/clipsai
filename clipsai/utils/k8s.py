"""
Defines Kubernetes persistent volume claim download directory.
"""
# standard library imports
import os

K8S_PVC_DIR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
