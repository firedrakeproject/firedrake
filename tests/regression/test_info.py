from firedrake import *


def test_info():
    info("This is normal text")
    info_red("This is red text")
    info_green("This is green text")
    info_blue("This is blue text")

    set_log_level(ERROR)
    info("You should not see this")
    log(ERROR, "You should see this error message")
    log(WARNING, "You should NOT see this warning")
