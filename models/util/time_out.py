from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError


# forked from Discussion board


def call_with_timeout(seconds, f, *args, **kwargs):
    # Define a function that forwards the arguments but times out
    @timeout(seconds)
    def f_timeout(*args, **kwargs):
        return f(*args, **kwargs)

    # Call the timeout wrapper function
    try:
        return f_timeout(*args, **kwargs)
    except TimeoutError as e:
        print()
        print("Error: function timed out after %f seconds with "
              "args=%s, kwargs=%s" % (seconds, args, kwargs))
