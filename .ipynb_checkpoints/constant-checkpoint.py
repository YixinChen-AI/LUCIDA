from label import mapping
def get_bones_indexes():
    bones = [7,8,13,14] + list(range(27,31)) + list(range(49,76)) + list(range(81,105)) + [110,111]
    new_mapping = {key:mapping[key] for key in mapping if key in bones}
    return new_mapping