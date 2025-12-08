def CheckMissingModules(moduleNames: list[str]) -> list[str]:
    """
    Check for missing Python modules from a given list of module names.

    Args:
        moduleNames (list[str]): A list of module names to check.
    Returns:
        list[str]: A list of missing module names.
    """
    missingModules: list[str] = []
    for moduleName in moduleNames:
        try:
            __import__(moduleName)
        except ImportError:
            missingModules.append(moduleName)
    return missingModules

def InstallMissingModules(missingModules: list[str]) -> None:
    """
    Install missing Python modules using pip.

    Args:
        missingModules (list[str]): A list of missing module names to install.
    """
    import subprocess
    import sys

    for moduleName in missingModules:
        subprocess.check_call([sys.executable, "-m", "pip", "install", moduleName])

def CheckPythonVersion(version: tuple[int, int]) -> bool:
    """
    Check if the current Python version matches the specified version.
    Args:
        version (tuple[int, int]): A tuple representing the required Python version (major, minor).
    Returns:
        bool: True if the current Python version matches the specified version, False otherwise.
    """
    import sys
    currentVersion: tuple[int, int] = sys.version_info[:2]
    print(f"Current Python version: {currentVersion[0]}.{currentVersion[1]}")
    return currentVersion == version

def Main() -> None:
    minimumVersion: tuple[int, int] = (3, 10)
    if not CheckPythonVersion(minimumVersion):
        print(f"Python {minimumVersion[0]}.{minimumVersion[1]} is required to run this script.")
        return
    modulesToCheck: list[str] = ["numpy", "mesh_to_sdf", "pandas", "trimesh", "scikit-image", "pyrender", "psutil"]
    missingModules: list[str] = CheckMissingModules(modulesToCheck)
    yesOrNo: bool = False
    if missingModules:
        print(f"The following modules are missing: {', '.join(missingModules)}")
        userInput: str = input("Do you want to install the missing modules? (Y/n): ").strip().lower()
        yesOrNo = userInput == 'y' or userInput == ''
    if yesOrNo:
        InstallMissingModules(missingModules)
        print("Missing modules have been installed.")
    else:
        print("Exiting without installing missing modules.")

if __name__ == "__main__":
    Main()
else:
    print("This script is intended to be run as the main module.")