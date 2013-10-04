#!/bin/sh
STDERR=$($* 2>&1 > /dev/null); RETCODE=$?
if [ $RETCODE -ne 0 ]
then
    echo "----" >&2
    echo "Command was: " >&2
    echo "$*" >&2
    echo "----" >&2
    echo "Error was: " >&2
    echo "$STDERR" >&2
    echo "----" >&2
fi
exit $RETCODE
