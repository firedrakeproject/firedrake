#!/usr/bin/env python
from __future__ import with_statement
import logging
import optparse
import os
import os.path
import re
import shutil
import subprocess
import sys
import itertools
import time

logger = logging.getLogger()

env_bin_dir = 'bin'
if sys.platform == 'win32':
    env_bin_dir = 'Scripts'

class UserError(Exception):
    pass

def in_directory(path, tree):
    "Make both paths absolute."
    directory = os.path.join(os.path.realpath(tree), '')
    file = os.path.realpath(path)

    #return true, if the common prefix of both is equal to directory
    #e.g. /a/b/c/d.rst and directory is /a/b, the common prefix is /a/b
    return os.path.commonprefix([path, tree]) == tree


def _dirmatch(path, matchwith):
    """Check if path is within matchwith's tree.

    >>> _dirmatch('/home/foo/bar', '/home/foo/bar')
    True
    >>> _dirmatch('/home/foo/bar/', '/home/foo/bar')
    True
    >>> _dirmatch('/home/foo/bar/etc', '/home/foo/bar')
    True
    >>> _dirmatch('/home/foo/bar2', '/home/foo/bar')
    False
    >>> _dirmatch('/home/foo/bar2/etc', '/home/foo/bar')
    False
    """
    matchlen = len(matchwith)
    if (path.startswith(matchwith)
        and path[matchlen:matchlen + 1] in [os.sep, '']):
        return True
    return False


def _virtualenv_sys(venv_path):
    "obtain version and path info from a virtualenv."
    executable = os.path.join(venv_path, env_bin_dir, 'python')
    my_env = os.environ.copy()
    # Must use "executable" as the first argument rather than as the
    # keyword argument "executable" to get correct value from sys.path
    p = subprocess.Popen([executable,
                          '-c', 'import sys;'
                          'print (sys.version[:3]);'
                          'print ("\\n".join(sys.path));'],
                         env=my_env,
                         stdout=subprocess.PIPE)
    stdout, err = p.communicate()
    logger.debug("_venv_sys stdout %s + return code: %s" % (stdout, err))
    assert not p.returncode and stdout
    lines = stdout.decode('utf-8').splitlines()
    return lines[0], filter(bool, lines[1:])

def grep_and_sed(src_dir, dst_dir):
    """ Grep and Sed: any mentions of src_dir in files in dst_dir are changed to dst_dir """ 
    grep = subprocess.Popen(['grep', '-Ilre', src_dir, dst_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t = ''.join(["s|", src_dir, "|", dst_dir, "|g"])
    sed  = subprocess.Popen(['xargs', 'sed', '-i', t], stdin=grep.stdout, stdout=subprocess.PIPE)
    stdout, errcode = sed.communicate()
    logger.debug("stdout from sed: %s" % stdout)
    logger.debug("errcode from sed: %s" % errcode)
    

def clone_virtualenv(src_dir, dst_dir, do_copy=True):
    """ Central Function: Move the whole directory tree, except .pyc files """
    
    # It is possible that you want to go through the process more than once, but the second time, you don't want to do the copy.
    # This is risky, as if do_copy is False, you will skip the checks on file existence etc.
    if do_copy==True:
        if not os.path.exists(src_dir):
            raise UserError('src dir %r does not exist' % src_dir)
        if os.path.exists(dst_dir):
            raise UserError('dest dir %r exists' % dst_dir)
        logger.info('cloning virtualenv \'%s\' => \'%s\'...' % (src_dir, dst_dir))
        shutil.copytree(src_dir, dst_dir, symlinks=True, ignore=shutil.ignore_patterns('*.pyc'))
    else:
        logger.info('NOT copying files.')

    # If using a chroot jail, you can end-up in a situation where you can never invoke the interpreter to determine the version
    # Ie, copying into the jail, the path would be /test/home/firedrake and invokable
    # But, from inside the jail, /test/home/firedrake doesn't exist, it's just /home/firedrake
    if do_copy==False:
        pybinre = re.compile('python?([0-9]+(\.[0-9])$)')
        root, dirs, files = next(os.walk(os.path.join(dst_dir, env_bin_dir)))
        for f_ in files:
            if pybinre.match(f_):
                version = ''.join(f_.replace("python", "", 1))
    else:
        version, sys_path = _virtualenv_sys(dst_dir)

    grep_and_sed(src_dir, dst_dir)
        
    logger.info('Fixing scripts in bin/...')
    fixup_scripts(src_dir, dst_dir, version)
    version, sys_path = _virtualenv_sys(dst_dir)
    has_old = lambda s: any(i for i in s if in_directory(i, src_dir))

    if has_old(sys_path):
        # only need to fix stuff in sys.path if we have old
        # paths in the sys.path of new python env. right?
        logger.info('Fixing paths in sys.path...')
        fixup_syspath_items(sys_path, src_dir, dst_dir)
    v_sys = _virtualenv_sys(dst_dir)
    remaining = has_old(v_sys[1])  
    assert not remaining, v_sys
    fix_symlink_if_necessary(src_dir, dst_dir)

    # There are some cases where, possibly due to CoW, you can end up modifying files in the source tree, as well as the dest
    # tree without knowing about it.
    # This parses back through the source and tries to undo any bad changes.
    if do_copy==True:
        version, v_sys = _virtualenv_sys(src_dir)
        fixup_syspath_items(v_sys, dst_dir, src_dir)

def fix_symlink_if_necessary(src_dir, dst_dir):
    #sometimes the source virtual environment has symlinks that point to itself
    #one example is $OLD_VIRTUAL_ENV/local/lib points to $OLD_VIRTUAL_ENV/lib
    #this function makes sure
    #$NEW_VIRTUAL_ENV/local/lib will point to $NEW_VIRTUAL_ENV/lib
    #usually this goes unnoticed unless one tries to upgrade a package though pip, so this bug is hard to find.
    logger.info("scanning for internal symlinks that point to the original virtual env")
    for dirpath, dirnames, filenames in os.walk(dst_dir):
        for a_file in itertools.chain(filenames, dirnames):
            full_file_path = os.path.join(dirpath, a_file)
            if os.path.islink(full_file_path):
                target = os.path.realpath(full_file_path)
                if target.startswith(src_dir):
                    new_target = target.replace(src_dir, dst_dir)
                    logger.debug('fixing symlink in {}'.format(full_file_path))
                    os.remove(full_file_path)
                    os.symlink(new_target, full_file_path)


def fixup_scripts(old_dir, new_dir, version, rewrite_env_python=False):
    bin_dir = os.path.join(new_dir, env_bin_dir)
    root, dirs, files = next(os.walk(bin_dir))
    pybinre = re.compile('pythonw?([0-9]+(\.[0-9]+(\.[0-9]+)?)?)?$')
    for file_ in files:
        filename = os.path.join(root, file_)
        if file_ in ['python', 'python%s' % version, 'activate_this.py']:
            continue
        elif file_.startswith('python') and pybinre.match(file_):
            # ignore other possible python binaries
            continue
        elif file_.endswith('.pyc'):
            # ignore compiled files
            continue
        elif file_ == 'activate' or file_.startswith('activate.'):
            fixup_activate(os.path.join(root, file_), old_dir, new_dir)
        elif os.path.islink(filename):
            fixup_link(filename, old_dir, new_dir)
        elif os.path.isfile(filename):
            fixup_script_(root, file_, old_dir, new_dir, version,
                rewrite_env_python=rewrite_env_python)


def fixup_script_(root, file_, old_dir, new_dir, version, rewrite_env_python=False):
    
    old_shebang = '#!%s/bin/python' % os.path.normcase(os.path.abspath(old_dir))
    new_shebang = '#!%s/bin/python' % os.path.normcase(os.path.abspath(new_dir))
    env_shebang = '#!/usr/bin/env python'

    filename = os.path.join(root, file_)
    with open(filename, 'rb') as f:
        if f.read(2) != b'#!':
            # no shebang
            return
        f.seek(0)
        lines = f.readlines()

    if not lines:
        # warn: empty script
        return

    def rewrite_shebang(version=None):
        logger.debug('fixing %s' % filename)
        shebang = new_shebang
        if version:
            shebang = shebang + version
        shebang = (shebang + '\n').encode('utf-8')
        with open(filename, 'wb') as f:
            f.write(shebang)
            f.writelines(lines[1:])

    try:
        bang = lines[0].decode('utf-8').strip()
    except UnicodeDecodeError:
        # binary file
        return

    if not bang.startswith('#!'):
        return
    elif bang == old_shebang:
        rewrite_shebang()
    elif (bang.startswith(old_shebang)
          and bang[len(old_shebang):] == version):
        rewrite_shebang(version)
    elif rewrite_env_python and bang.startswith(env_shebang):
        if bang == env_shebang:
            rewrite_shebang()
        elif bang[len(env_shebang):] == version:
            rewrite_shebang(version)
    else:
        # can't do anything
        return


def fixup_activate(filename, old_dir, new_dir):
    logger.debug('fixing %s' % filename)
    with open(filename, 'rb') as f:
        data = f.read().decode('utf-8')

    data = data.replace(old_dir, new_dir)
    with open(filename, 'wb') as f:
        f.write(data.encode('utf-8'))


def fixup_link(filename, old_dir, new_dir, target=None):
    logger.debug('Fixing link: %s' % filename)
    if target is None:
        target = os.readlink(filename)

    origdir = os.path.dirname(os.path.abspath(filename)).replace(
        new_dir, old_dir)
    if not os.path.isabs(target):
        target = os.path.abspath(os.path.join(origdir, target))
        rellink = True
    else:
        rellink = False

    if in_directory(target, old_dir):
        if rellink:
            # keep relative links, but don't keep original in case it
            # traversed up out of, then back into the venv.
            # so, recreate a relative link from absolute.
            target = target[len(origdir):].lstrip(os.sep)
        else:
            target = target.replace(old_dir, new_dir, 1)

    # else: links outside the venv, replaced with absolute path to target.
    _replace_symlink(filename, target)


def _replace_symlink(filename, newtarget):
    tmpfn = "%s.new" % filename
    os.symlink(newtarget, tmpfn)
    os.rename(tmpfn, filename)


def fixup_syspath_items(syspath, old_dir, new_dir):
    logger.debug('Traversing syspath looking for anything which may point to old path')
    for path in syspath:
        if not os.path.isdir(path):
            continue
        path = os.path.normcase(os.path.abspath(path))
        if in_directory(path, old_dir):
            # print "Matched %s %s " % (path, old_dir)
            path = path.replace(old_dir, new_dir, 1)
            # print "Path is now %s " % syspath
            if not os.path.exists(path):
                continue
        elif not in_directory(path, new_dir):
            # print "Failed match %s %s" % (path, new_dir)
            continue
        root, dirs, files = next(os.walk(path))
        for file_ in files:
            filename = os.path.join(root, file_)
            if filename.endswith('.pth'):
                fixup_pth_file(filename, old_dir, new_dir)
            elif filename.endswith('.egg-link'):
                fixup_egglink_file(filename, old_dir, new_dir)


def fixup_pth_file(filename, old_dir, new_dir):
    logger.debug('Fixing pth file: %s' % filename)
    with open(filename, 'rb') as f:
        lines = f.readlines()
    f.close()
    has_change = False
    for num, line in enumerate(lines):
        line = line.decode('utf-8')
        if not line or line.startswith('#') or line.startswith('import '):
            continue
        elif in_directory(line, old_dir):
            lines[num] = line.replace(old_dir, new_dir, 1)
            lines[num] = lines[num].encode('utf-8')
            has_change = True
    if has_change:
        with open(filename, 'wb') as f:
            f.writelines(lines)
        f.close()


def fixup_egglink_file(filename, old_dir, new_dir):
    logger.debug('Fixing egg-link file: %s' % filename)
    with open(filename, 'rb') as f:
        link = f.read().decode('utf-8').strip()
    f.close()
    if in_directory(link, old_dir):
        link = link.replace(old_dir, new_dir, 1)
        with open(filename, 'wb') as f:
            if link.find('\n'):
                link = link.encode('utf-8')
            else:
                link = (link + '\n').encode('utf-8')
            f.write(link)
        f.close()


def main():
    parser = optparse.OptionParser("usage: %prog [options]" " /path/to/existing/venv /path/to/cloned/venv")
    parser.add_option('-v', action="count", dest='verbose', default=False, help='verbosity')
    parser.add_option('-d', action="store_false", dest='copy', default=True, help='copy?')
    options, args = parser.parse_args()
    try:
        old_dir, new_dir = args
    except ValueError:
        parser.error("not enough arguments given.")
    old_dir = os.path.normpath(os.path.abspath(old_dir))
    new_dir = os.path.normpath(os.path.abspath(new_dir))
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(2, options.verbose)]
    logging.basicConfig(level=loglevel, format='%(message)s')
    try:
        print options.copy
        clone_virtualenv(old_dir, new_dir, options.copy)
    except UserError:
        e = sys.exc_info()[1]
        parser.error(str(e))


if __name__ == '__main__':
    main()
