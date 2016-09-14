import subprocess

def svnversion():
    p = subprocess.Popen("svnversion", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    return stdout

def svnstatus():
    p = subprocess.Popen("svn status", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    return stdout

def svndiff():
    p = subprocess.Popen("svn diff", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    return stdout