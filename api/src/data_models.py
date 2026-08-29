def validate_dict_for_keys(dict_object, keys, loc):
    validation_errors = []

    for key in keys:
        if dict_object.get(key, None) is None:
            validation_errors.append({
                'loc': loc,
                'error': f"{key} not provided."
            })
    
    return validation_errors


def validate_segment_single_slice_request_headers(request_headers):
    required_headers = [
        "X-IML-RELATIVE_VOLUME_PATH",
        "X-IML-SLICE_TYPE",
        "X-IML-SLICE_INDEX",
    ]

    return validate_dict_for_keys(request_headers, required_headers, "Headers")


def validate_interactive_segment_single_slice_request_headers(request_headers):
    required_headers = [
        "X-IML-RELATIVE_VOLUME_PATH",
        "X-IML-SLICE_TYPE",
        "X-IML-SLICE_INDEX",
    ]

    return validate_dict_for_keys(request_headers, required_headers, "Headers")
