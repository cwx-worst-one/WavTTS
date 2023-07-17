'''
util method for queue operation.
'''


def safe_get(stop_queue, queue, default_data=(), timeout=1, retry=40):
    '''safe get for a queue.'''
    data = default_data
    while stop_queue.empty() and retry > 0:
        try:
            data = queue.get(block=True, timeout=timeout)
            break
        except Exception:
            pass
        retry -= 1
    return data


def safe_put(stop_queue, queue, data, timeout=0.5, retry=800000):
    '''
    safe put for a queue.
    default will wait 111 hours.
    '''
    while stop_queue.empty() and retry > 0:
        try:
            queue.put(data, block=True, timeout=timeout)
            return True
        except Exception:
            pass
        retry -= 1
    return False


def thread_safe_get(done_event, queue, default_data=(), timeout=1, retry=40):
    '''safe get for a queue.'''
    data = default_data
    while not done_event.is_set() and retry > 0:
        try:
            data = queue.get(block=True, timeout=timeout)
            break
        except Exception:
            pass
        retry -= 1
    return data


def thread_safe_put(done_event, queue, data, timeout=0.5, retry=800000):
    '''
    safe put for a queue.
    default will wait 111 hours.
    '''
    while not done_event.is_set() and retry > 0:
        try:
            queue.put(data, block=True, timeout=timeout)
            return True
        except Exception:
            pass
        retry -= 1
    return False
