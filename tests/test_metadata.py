from importlib.metadata import version

import csm_processor


def test_runtime_version_matches_distribution_metadata():
    assert csm_processor.__version__ == version("csm-processor")