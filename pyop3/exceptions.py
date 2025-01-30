# NOTE: I don't like that this file exists, exceptions should generally be defined where
# they occur IMO. Here we have generic ones though so I'm not sure.

class Pyop3Exception(Exception):
    pass


class DataValueError(ValueError, Pyop3Exception):
    pass
