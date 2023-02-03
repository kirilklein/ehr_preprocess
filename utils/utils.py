def prepend(input, prepend_token, slice_start=None, slice_end=None):
    return prepend_token + slice_wrapper(input, slice_start, slice_end)

def slice_wrapper(input, slice_start=None, slice_end=None):
    return input[slice_start:slice_end]