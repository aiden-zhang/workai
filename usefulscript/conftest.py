import pytest

def pytest_addoption(parser):
    parser.addoption("--out_node",default="",help="one of out node name")
    parser.addoption("--max_batch", default="",help="max execute batch,should be numberic if not it will set to 1")
    parser.addoption("--exec_batch", default="",help="execute batch,should be numberic if not it will set to 1")
    parser.addoption("--weight_share", default="0", help="it should be one of {0,1,2,3,01,23,0123}")
    parser.addoption("--file_name",default="",help="name of the serialize file")
    parser.addoption("--model_dir",default="",help="model files directory")
    parser.addoption("--loop",default="1",help="run loop count")
    parser.addoption("--random_input", default="2", help="random input data, 0: all ones, 1: pseudo random, 2: true random")
    parser.addoption("--slz", default="", help="serialize network to file name")
    parser.addoption("--deslz", default="", help="deserialize network from file name")
    parser.addoption("--use_tvm_plugin", default="False", help="if use tvm plugin")
    parser.addoption("--builder_flag", default="", help="any ones of {spm_alloc,}, seperated by '|'")

@pytest.fixture
def out_node(request):
    return request.config.getoption("--out_node")

@pytest.fixture
def max_batch(request):
    return request.config.getoption("--max_batch")

@pytest.fixture
def exec_batch(request):
    return request.config.getoption("--exec_batch")

@pytest.fixture
def weight_share(request):
    return request.config.getoption("--weight_share")

@pytest.fixture
def file_name(request):
    return request.config.getoption("--file_name")

@pytest.fixture
def model_dir(request):
    return request.config.getoption("--model_dir")

@pytest.fixture
def loop(request):
    return request.config.getoption("--loop")

@pytest.fixture
def random_input(request):
    return request.config.getoption("--random_input")

@pytest.fixture
def slz(request):
    return request.config.getoption("--slz")

@pytest.fixture
def deslz(request):
    return request.config.getoption("--deslz")

@pytest.fixture
def use_tvm_plugin(request):
    return request.config.getoption("--use_tvm_plugin")

@pytest.fixture
def builder_flag(request):
    return request.config.getoption("--builder_flag")
