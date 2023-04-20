def prepend(input, prepend_token, slice_start=None, slice_end=None):
    return prepend_token + slice_wrapper(input, slice_start, slice_end)

def slice_wrapper(input, slice_start=None, slice_end=None):
    return input[slice_start:slice_end]

def bin_bmi(bmi: float):
    bmi = float(bmi)
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    elif bmi < 35:
        return 'obese'
    elif bmi < 40:
        return 'extremely-obese'
    elif bmi < 100:
        return 'morbidly-obese'
    else:
        return None