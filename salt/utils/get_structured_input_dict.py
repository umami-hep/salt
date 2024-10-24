def get_structured_input_dict(input_dict, variable_map, global_object):
    structured_input_dict = {}
    for obj_type in [global_object]:  # should be extended to include non-global objs ideally
        structured_input_dict[obj_type] = {}
        for i, var in enumerate(variable_map[obj_type]):
            structured_input_dict[obj_type][var] = input_dict[obj_type][:, i]
    return structured_input_dict
