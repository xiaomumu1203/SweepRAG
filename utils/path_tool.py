import os

#获取当前项目的根目录
def get_project_root() -> str:
    #获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    #获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    #获取当前项目的根目录
    project_root = os.path.dirname(current_dir)
    return project_root

#获取当前文件的绝对路径
def get_abs_path(relative_path:str) -> str:
    #获取当前文件的根目录
    project_root = get_project_root()
    #和文件名称拼接出绝对路径
    return os.path.join(project_root, relative_path)
