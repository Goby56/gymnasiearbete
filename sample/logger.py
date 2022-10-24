import datetime
posts = []

def println(*args, post_after=False, timestamp=False) -> None:
    """
    post_after = True -> posts message after the whole program is finished
    timestamp = True -> adds a timestamp to the message
    """
    time_stamp = f"[{datetime.datetime.now().strftime('%H:%M:%f')}] "
    message = f"{time_stamp*timestamp}"
    for arg in args: 
        message += f"{arg} "

    if post_after:
        posts.append(message)
    else: 
        print(message)

def log_function(post_after=False, timestamp=False) -> None:
    """
    log function is meant to be used as a decorator.
    post_after = True -> posts message after the whole program is finished
    timestamp = True -> adds a timestamp to the message
    example: [timestamp] function: test() was called with return value None, arguments (5, 1), and key-word arguments: {}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return_value = func(*args, **kwargs)
            println(f"function: {func.__name__}() was called with return value: {return_value}, arguments: {args}, and key-word arguments {kwargs}",
                    post_after=post_after, timestamp=timestamp)
            return return_value
        return wrapper
    return decorator
    
def print_post() -> None:
    """
    print_post is called at the end of the program regardless of errors.
    it's used to print messages that are speceified to be prtinted at the end of the program.
    """
    for post in posts:
        print(post)

if __name__ == "__main__":
    @log_function(post_after=True, timestamp=True)
    def test(a, b):
        return a + b
    test(1, 2)
    print_post()
