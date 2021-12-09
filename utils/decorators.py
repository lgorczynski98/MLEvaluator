import collections
import functools
import time
import os
import logging
from collections.abc import Iterable

def debug(func):
    ''' Decorator for debug purposes. 
        Display information about run funciton and its parameters.
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(arg) for arg in args if not isinstance(arg, Iterable)]
        kwargs_repr = [f'{k}={v!r}' for k, v in kwargs.items()]
        args_kwargs = ', '.join(args_repr + kwargs_repr)
        logging.info(f'### Calling {func.__name__}({args_kwargs}) ###')
        val = func(*args, **kwargs)
        logging.info(f'### End call {func.__name__}({args_kwargs}) ###')
        return val
    return wrapper

def log(func):
    ''' Decorator informint about what function was called '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f'### Calling {func.__name__} ###')
        val = func(*args, **kwargs)
        logging.info(f'### End call {func.__name__} ###')
        return val
    return wrapper

def timer(func):
    ''' Timer decorator. Counts funciton call time. '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        val = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f'### {func.__name__} finished in {run_time:4f} seconds ###')
        return val
    return wrapper

def save_plot(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'file_name' in kwargs:
            timestamp = int(time.time())
            file_name = f"{kwargs.pop('file_name')}_{timestamp}"
        else:
            file_name = None
        
        if 'data_dim' in kwargs:
            data_dim = kwargs.pop('data_dim')
        else:
            data_dim = 2

        plt = func(*args, **kwargs)
        if not file_name is None:
            dest = os.path.join('plots')
            file_path = os.path.join(dest, file_name)

            if data_dim == 2:
                plt.savefig(f'{file_path}.png', format='png', bbox_inches='tight')
            elif data_dim == 3:
                plt.write_html(f'{file_path}.html')

            logging.info(f'Saved {file_name}')
            return
        plt.show()
    return wrapper