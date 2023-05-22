import os
import importlib
from types import ModuleType

class LazyLoader(ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    `contrib`, and `ffmpeg` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.
    """

    def __init__(self, local_name, parent_module_globals, name):
        '''init'''
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        # pylint: disable='super-with-arguments'
        super(LazyLoader, self).__init__(name)

    def _load(self):
        '''load'''
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        '''__getattr__'''
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        '''__dir__'''
        module = self._load()
        return dir(module)


# pylint: disable='too-many-return-statements'
def get_hdfs_host():
    '''get hdfs host'''
    arnold_base_dir = os.environ.get('ARNOLD_BASE_DIR', '')
    if arnold_base_dir.startswith('hdfs://harunava'):
        return 'hdfs://harunava'
    if arnold_base_dir.startswith('hdfs://harunaoci'):
        return 'hdfs://harunaoci'
    if arnold_base_dir.startswith('hdfs://haruna'):
        return 'hdfs://haruna'
    if os.environ.get('ARNOLD_WORKSPACE_CLUSTER_NAME') == 'candy-maliva':
        return 'hdfs://harunava'
    try:
        # pylint: disable='import-outside-toplevel'
        import xml.etree.ElementTree as ET

        tree = ET.parse('/opt/tiger/yarn_deploy/hadoop/conf/core-site.xml')
        root = tree.getroot()
        for child in root:
            if child.tag == 'property' and child[0].text == 'fs.defaultFS':
                return child[1].text
    except Exception:
        return 'hdfs://haruna'
    return 'hdfs://haruna'


# pylint: disable='import-outside-toplevel'
def get_hdfs_block_size():
    '''get hdfs block size'''
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse('/opt/tiger/yarn_deploy/hadoop/conf/hdfs-site.xml')
        root = tree.getroot()
        for child in root:
            if child.tag == 'property' and child[0].text == 'dfs.block.size':
                return int(child[1].text)
    except Exception:
        pass
    return 134217728


def get_parquet_file_handle(url):
    '''get parquet file handle'''
    pq = LazyLoader('pq', globals(), 'pyarrow.parquet')
    pf = LazyLoader('pf', globals(), 'pyarrow.fs')
    if url.startswith("hdfs"):
        fs = pf.HadoopFileSystem(host=get_hdfs_host(), port=0, buffer_size=get_hdfs_block_size())
        f = fs.open_input_file(url)
    else:
        # pylint: disable='consider-using-with'
        f = open(url, 'rb')
        fs = pf.LocalFileSystem()
    parquet_file = pq.ParquetFile(f)
    return parquet_file, fs, f


def get_parquet_file_info(url):
    '''get parquet file info'''
    parquet_file, _, file_handle = get_parquet_file_handle(url)
    num_row_groups = parquet_file.metadata.num_row_groups
    num_rows = parquet_file.metadata.num_rows
    meta_data = dict(num_row_groups=num_row_groups, num_rows=num_rows)
    file_handle.close()
    return meta_data