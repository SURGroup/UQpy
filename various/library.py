# Read parameter file
import os

def read_param_file(filename):
    """
    Reads input file, interprets it and runs necessary code
    :param filename:
    :return:
    """
    names = []
    parameters = []
    dim = 0
    distribution = []
    save_name = []
    save_format = []
    method = None
    number = None

    with open(filename, "r") as file:
        for row in [line.split() for line in file if not line.strip().startswith('#')]:
            dim += 1
            names.append(row[0])
            parameters.append([float(row[2]), float(row[3])])
            distribution.append(row[1])
            if dim == 1:
                method = read_method(row[4])
                number = read_number(row[5])
                save_name = row[6]
                save_format = row[7]

        cwd = os.getcwd()    # current directory
    '''
    if method is None:
        return
    
    if number is None:
        return
    '''

    return {'dim': dim, 'names': names, 'distribution': distribution, 'parameters': parameters, 'method': method,
            'number': number, 'save_name': save_name, 'save_format': save_format}


########################################################################################################################
def read_method(s):
    """
    This function reads in a string and checks if it is a supported sampling method. If not, it displays an error
    message.
    :param s: string which should be the sampling method keyword
    :return: The string s if it a supported method or None if the string s is not a supported sampling method
    """
    planned_methods_list = ['mcs', 'lhs', 'quasi_mcs', 'lss', 'dt', 'voronoi', 'mcmc', 'stratified']
    implemented_methods_list = ['mcs', 'lhs']
    
    check_method_string = isinstance(s, str)  # This checks if the method is a string
    if not check_method_string:
        return
    else:
        # check_is_number = is_number(out['method'])
        # if check_is_number:
        #     print('The method should not be a number.')
        #     return
        #
        # check_has_number = has_number(out['method'])
        # if check_has_number:
        #     print('The method should not contain numbers.')
        #     return

        if s.lower() not in planned_methods_list:
            raise SyntaxError('Invalid sampling method or this method is not supported.')
            # print("\n**ERROR**\nInvalid sampling method keyword or this sampling method is not supported. "
            #       "Please choose a different sampling strategy.\n"
            #       "\nUse one of the following sampling keywords: 'mcs', 'lhs', 'quasi_mcs', 'lss', 'dt', 'voronoi', "
            #       "'mcmc', 'stratified'")
            # return
        elif s.lower() not in implemented_methods_list:
            raise NotImplementedError('This method will be implemented soon.')
        else:
            return s
########################################################################################################################


def read_number(s):
    """
    This function checks if the number is an integer or not. If not, it displays an error message.
    :param s: string which should be an integer
    :return: An integer, which is the string s converted to int type if integer, None otherwise
    """
    if not is_integer(s):  # Check if the number is an integer
        print('\n**ERROR**\nThe number of samples should be an integer.')
        return
    else:
        return int(s)
########################################################################################################################


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
########################################################################################################################

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

########################################################################################################################

def has_number(input_string):
    return any(char.isdigit() for char in input_string)
########################################################################################################################


