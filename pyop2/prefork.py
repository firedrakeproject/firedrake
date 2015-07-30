# Taken from Andreas Kloeckner's pytools package
# https://github.com/inducer/pytools
# MIT License

"""OpenMPI, once intialized, prohibits forking. This helper module
allows the forking of *one* helper child process before OpenMPI
initializaton that can do the forking for the fork-challenged
parent process.

Since none of this is MPI-specific, it got parked in pytools.
"""
from __future__ import absolute_import





class ExecError(OSError):
    pass




class DirectForker:
    @staticmethod
    def call(cmdline, cwd=None):
        from subprocess import call
        try:
            return call(cmdline, cwd=cwd)
        except OSError as e:
            raise ExecError("error invoking '%s': %s"
                    % ( " ".join(cmdline), e))

    @staticmethod
    def call_capture_stdout(cmdline, cwd=None):
        from subprocess import Popen, PIPE
        try:
            return Popen(cmdline, cwd=cwd, stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()[0]
        except OSError as e:
            raise ExecError("error invoking '%s': %s"
                    % ( " ".join(cmdline), e))

    @staticmethod
    def call_capture_output(cmdline, cwd=None, error_on_nonzero=True):
        """
        :returns: a tuple (return code, stdout_data, stderr_data).
        """
        from subprocess import Popen, PIPE
        try:
            popen = Popen(cmdline, cwd=cwd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            stdout_data, stderr_data = popen.communicate()
            if error_on_nonzero and popen.returncode:
                raise ExecError("status %d invoking '%s': %s"
                        % (popen.returncode, " ".join(cmdline), stderr_data))
            return popen.returncode, stdout_data, stderr_data
        except OSError as e:
            raise ExecError("error invoking '%s': %s"
                    % ( " ".join(cmdline), e))



def _send_packet(sock, data):
    from struct import pack
    from six.moves.cPickle import dumps

    packet = dumps(data)

    sock.sendall(pack("I", len(packet)))
    sock.sendall(packet)

def _recv_packet(sock, who="Process", partner="other end"):
    from struct import calcsize, unpack
    size_bytes_size = calcsize("I")
    size_bytes = sock.recv(size_bytes_size)

    if len(size_bytes) < size_bytes_size:
        from warnings import warn
        warn("%s exiting upon apparent death of %s" % (who, partner))

        raise SystemExit

    size, = unpack("I", size_bytes)

    packet = b""
    while len(packet) < size:
        packet += sock.recv(size)

    from six.moves.cPickle import loads
    return loads(packet)




def _fork_server(sock):
    import signal
    # ignore keyboard interrupts, we'll get notified by the parent.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    quitflag = [False]

    def quit():
        quitflag[0] = True

    funcs = {
    "quit": quit,
    "call": DirectForker.call,
    "call_capture_stdout": DirectForker.call_capture_stdout,
    "call_capture_output": DirectForker.call_capture_output,
    }

    try:
        while not quitflag[0]:
            func_name, args, kwargs = _recv_packet(sock, 
                    who="Prefork server", partner="parent")

            try:
                result = funcs[func_name](*args, **kwargs)
            except Exception as e:
                _send_packet(sock, ("exception", e))
            else:
                _send_packet(sock, ("ok", result))
    finally:
        sock.close()

    import os
    os._exit(0)





class IndirectForker:
    def __init__(self, server_pid, sock):
        self.server_pid = server_pid
        self.socket = sock

    def _remote_invoke(self, name, *args, **kwargs):
        _send_packet(self.socket, (name, args, kwargs))
        status, result = _recv_packet(self.socket, 
                who="Prefork client", partner="prefork server")

        if status == "exception":
            raise result
        elif status == "ok":
            return result

    def _quit(self):
        self._remote_invoke("quit")
        from os import waitpid
        waitpid(self.server_pid, 0)

    def call(self, cmdline, cwd=None):
        return self._remote_invoke("call", cmdline, cwd)

    def call_capture_stdout(self, cmdline, cwd=None):
        return self._remote_invoke("call_capture_stdout", cmdline, cwd)

    def call_capture_output(self, cmdline, cwd=None, error_on_nonzero=True):
        return self._remote_invoke("call_capture_output", cmdline, cwd, 
                error_on_nonzero)




def enable_prefork():
    if isinstance(forker[0], IndirectForker):
        return

    from socket import socketpair
    s_parent, s_child = socketpair()

    from os import fork
    fork_res = fork()

    if fork_res == 0:
        # child
        s_parent.close()
        _fork_server(s_child)
    else:
        s_child.close()
        forker[0] = IndirectForker(fork_res, s_parent)

        import atexit
        atexit.register(forker[0]._quit)




forker = [DirectForker()]

def call(cmdline, cwd=None):
    return forker[0].call(cmdline, cwd)

def call_capture_stdout(cmdline, cwd=None):
    from warnings import warn
    warn("call_capture_stdout is deprecated: use call_capture_output instead",
            stacklevel=2)
    return forker[0].call_capture_stdout(cmdline, cwd)

def call_capture_output(cmdline, cwd=None, error_on_nonzero=True):
    return forker[0].call_capture_output(cmdline, cwd, error_on_nonzero)
