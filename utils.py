import os
import sys
import platform
import logging
from IPython import get_ipython


# TODO: there needs to be a yaml file to set up a folder structure, hardcoding here is not good :)
# Question: Or should this be part of the momics package?
def init_setup():
    # First solve if IPython
    if is_ipython():
        ## For running at GColab, the easiest is to clone and then pip install some deps
        setup_ipython()

    else:
        setup_local()


def setup_local():
    # I do not install the package via pip install -e, I rather add the path to the package to the sys.path
    # faster prototyping of the momics package
    if platform.system() == "Linux":
        print("Platform: local Linux")
        sys.path.append("/home/davidp/python_projects/marine_omics/marine-omics")
    elif platform.system() == "Windows":
        print("Platform: local Windows")
        sys.path.append(
            "C:/Users/David Palecek/Documents/Python_projects/marine_omics/marine-omics"
        )
    else:
        raise NotImplementedError

def install_common_remote_packages():
    try:
        os.system("git clone https://github.com/palec87/marine-omics.git")
        print(f"Repository marine-omics cloned")
    except OSError as e:
        print(f"An error occurred while cloning the repository: {e}")

    try:
        os.system("pip install git+https://github.com/palec87/marine-omics.git")
        print(f"momics installed")
    except OSError as e:
        print(f"An error occurred while installing momics: {e}")

    try:
        os.system("pip install panel hvplot")
        print(f"panel and hvplot installed")
    except OSError as e:
        print(f"An error occurred while installing panel and hvplot: {e}")


def setup_ipython():
    """
    Setup the IPython environment.

    This function installs the momics package and other dependencies for the IPython environment.
    """
    if "google.colab" in str(get_ipython()):
        print("Google Colab")

        # Install ngrok for hosting the dashboard
        try:
            os.system("pip install pyngrok --quiet")
            print("ngrok installed")
        except OSError as e:
            print(f"An error occurred while installing ngrok: {e}")

        # Install the momics package
        install_common_remote_packages()

    elif "zmqshell" in str(get_ipython()) and "conda" in sys.prefix:  # binder
        print("Binder")
        install_common_remote_packages()
    else:
        # assume local jupyter server which has all the dependencies installed (because I do not do conda)
        # TODO: this is not general
        setup_local()

def is_ipython():
    # This is for the case when the script is run from the Jupyter notebook
    if "ipykernel" not in sys.modules:
        print("Not IPython setup")
        return False
    
    from IPython import get_ipython
    return True



def get_notebook_environment():
    """
    Determine if the notebook is running in VS Code or JupyterLab.

    Returns:
        str: The environment in which the notebook is running ('vscode', 'jupyterlab', or 'unknown').
    """
    # Check for VS Code environment variable
    if 'VSCODE_PID' in os.environ:
        return 'vscode'
    
    elif "JPY_SESSION_NAME" in os.environ:
        return 'jupyterlab'
    else:
        return 'unknown'


FORMAT = "%(levelname)s | %(name)s | %(message)s"  # for logger
def reconfig_logger(format=FORMAT, level=logging.INFO):
    """(Re-)configure logging"""
    logging.basicConfig(format=format, level=level, force=True)
    logging.debug("Logging.basicConfig completed successfully")
