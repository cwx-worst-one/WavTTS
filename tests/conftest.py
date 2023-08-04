import os
import pathlib
import shutil

import pytest
import torch


@pytest.fixture
def tests_root_dir():
    return os.path.dirname(__file__)


@pytest.fixture
def project_dir():
    return pathlib.Path(__file__).parent.parent

@pytest.fixture(scope="session")
def original_global_datadir():
    return pathlib.Path(os.path.realpath(__file__)).parent / "data"


def prep_global_datadir(tmp_path_factory, original_global_datadir):
    temp_dir = tmp_path_factory.mktemp("data") / "datadir"
    shutil.copytree(original_global_datadir, temp_dir)
    return temp_dir


@pytest.fixture(scope="session")
def session_global_datadir(tmp_path_factory, original_global_datadir):
    return prep_global_datadir(tmp_path_factory, original_global_datadir)


@pytest.fixture(scope="module")
def module_global_datadir(tmp_path_factory, original_global_datadir):
    return prep_global_datadir(tmp_path_factory, original_global_datadir)


@pytest.fixture(scope="function")
def global_datadir(tmp_path_factory, original_global_datadir):
    return prep_global_datadir(tmp_path_factory, original_global_datadir)


def pytest_addoption(parser):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(0)
    parser.addoption("--device", action="store", default=device)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value])


def pytest_configure(config):
    config.addinivalue_line("markers", "disable: mark tests as be disabled")
    config.addinivalue_line("markers", "benchmark: mark tests as benchmark cases")


collect_ignore = []
