from csm_processor import bin_csm, log_freq_bin
from csm_processor.log_binning import (
    bin_csm as bin_csm_impl,
    log_freq_bin as log_freq_bin_impl,
)


def test_public_api_exports_binning_helpers():
    assert bin_csm is bin_csm_impl
    assert log_freq_bin is log_freq_bin_impl