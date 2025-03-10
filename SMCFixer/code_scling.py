import subprocess
import json
import re

def capture_ast_from_command(filename):
    result = subprocess.run(['python3', '-m', 'solidity_parser', 'parse', filename],
                            capture_output=True, text=True)
    return result.stdout

def normalize_json(input_string):
    normalized_str = re.sub(r"\'", '"', input_string)
    normalized_str = re.sub(r'\bTrue\b', 'true', normalized_str)
    normalized_str = re.sub(r'\bFalse\b', 'false', normalized_str)
    normalized_str = re.sub(r'\bNone\b', 'null', normalized_str)
    return normalized_str

def remove_solidity_keywords(code_line):
    solidity_keywords = [
        'return', 'address', 'uint', 'uint160', 'int256', 'bool', 'string',
        'if', 'else', 'for', 'while', 'function', 'mapping', 'assembly',
        'public', 'private', 'internal', 'external', 'pure', 'view', 'payable', 'returns'
    ]
    tokens = re.split(r'\W+', code_line)  # Split by non-word characters
    filtered_tokens = [token for token in tokens if token and token not in solidity_keywords]
    return filtered_tokens

def extract_code_from_error_message(error_message):
    code_match = re.search(r'\|\n\s*\d*\s*\|\s*(.*?)\n\s*\|', error_message)
    if code_match:
        return code_match.group(1).strip()
    return None

def find_ast_nodes_by_elements(ast, elements):
    result_nodes = []
    def recursive_search(node, elements):
        if isinstance(node, dict):
            if any(node.get('name') == element for element in elements):
                result_nodes.append(node)
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    recursive_search(value, elements)
        elif isinstance(node, list):
            for item in node:
                recursive_search(item, elements)

    recursive_search(ast, elements)
    return result_nodes

def extract_global_variables(ast):
    global_vars = []
    for node in ast.get('children', []):
        if node.get('type') == 'ContractDefinition':
            for sub_node in node.get('subNodes', []):
                if sub_node.get('type') == 'VariableDeclaration':
                    var_type = sub_node.get('typeName').get('name')
                    var_name = sub_node.get('name')
                    global_vars.append(f"{var_type} {var_name};")
    return global_vars

def process_assembly_expression(expr):
    if expr['type'] == 'AssemblyExpression':
        args = [process_assembly_expression(arg) if 'type' in arg and arg['type'] == 'AssemblyExpression' else arg['value'] for arg in expr['arguments']]
        return f"{expr['functionName']}({', '.join(args)})"
    elif expr['type'] == 'AssemblyCall':
        args = [arg['value'] for arg in expr['arguments']]
        return f"{expr['functionName']}({', '.join(args)})"
    elif expr['type'] == 'DecimalNumber':
        return expr['value']
    elif expr['type'] == 'Identifier':
        return expr['name']
    return ''

def process_statement(stmt):
    slim_code = ""
    print(stmt['type'])
    print(stmt)

    if stmt['type'] == 'ExpressionStatement':
        expression = stmt['expression']
        if expression['type'] == 'BinaryOperation':
            left = expression['left'].get('name', '')
            right = expression['right'].get('value', expression['right'].get('name', ''))
            slim_code += f"        {left} {expression['operator']} {right};\n"
        elif expression['type'] == 'FunctionCall':
            callee = expression['expression'].get('name', '')
            args = ', '.join([arg.get('name', arg.get('value', '')) for arg in expression['arguments']])
            slim_code += f"        {callee}({args});\n"

    elif stmt['type'] == 'VariableDeclarationStatement':
        variable_name = stmt['variables'][0]['name']
        variable_type = stmt['variables'][0]['typeName']['name']
        initial_value = stmt['initialValue']

        if initial_value['type'] == 'UnaryOperation':
            operator = initial_value['operator']
            value = initial_value['subExpression']['number']
            slim_code += f"        {variable_type} {variable_name} = {operator}{value};\n"

    elif stmt['type'] == 'FunctionCall':
        function_name = stmt['expression']['name']
        arguments = ', '.join(
            [arg['name'] if 'name' in arg else arg['value'] for arg in stmt['arguments'][0]['arguments']])
        slim_code += f"        address({function_name}({arguments}));\n"

    elif stmt['type'] == 'ReturnStatement':
        return_value = stmt['expression'].get('name', stmt['expression'].get('value', ''))
        slim_code += f"        return {return_value};\n"

    elif stmt['type'] == 'IfStatement':
        condition_left = stmt['condition']['left'].get('name', '')
        condition_operator = stmt['condition'].get('operator', '')
        condition_right = stmt['condition']['right'].get('number', stmt['condition']['right'].get('name', ''))
        slim_code += f"        if ({condition_left} {condition_operator} {condition_right}) {{\n"
        for s in stmt['TrueBody']['statements']:
            slim_code += process_statement(s)
        slim_code += "        }\n        else {\n"
        for s in stmt['FalseBody']['statements']:
            slim_code += process_statement(s)
        slim_code += "        }\n"

    elif stmt['type'] == 'ForStatement':
        init_var = stmt['initExpression']['variables'][0].get('name', '')
        init_value = stmt['initExpression']['initialValue'].get('number', '')
        condition_left = stmt['conditionExpression']['left'].get('name', '')
        condition_operator = stmt['conditionExpression'].get('operator', '')
        condition_right = stmt['conditionExpression']['right']['expression'].get('name', '') + '.' + \
                          stmt['conditionExpression']['right'].get('memberName', '')
        increment = stmt['loopExpression']['expression']['subExpression'].get('name', '') + stmt['loopExpression'][
            'expression'].get('operator', '++')
        slim_code += f"        for ({init_var} = {init_value}; {condition_left} {condition_operator} {condition_right}; {increment}) {{\n"
        for s in stmt['body']['statements']:
            slim_code += process_statement(s)
        slim_code += "        }\n"

    elif stmt['type'] == 'InlineAssemblyStatement':
        slim_code += "        assembly {\n"
        for op in stmt['body']['operations']:
            if op['type'] == 'AssemblyAssignment':
                names = ", ".join([name['name'] for name in op['names']])
                expr = process_assembly_expression(op['expression'])
                slim_code += f"            {names} := {expr};\n"
        slim_code += "        }\n"

    return slim_code

def generate_slim_code_from_ast(ast, elements):
    slim_code = ""
    for node in ast.get('children', []):
        if node.get('type') == 'ContractDefinition':
            for sub_node in node.get('subNodes', []):
                if sub_node.get('type') == 'FunctionDefinition' and sub_node.get('name', '') in elements:
                    slim_code += f"function {sub_node['name']}("
                    params = sub_node.get('parameters', {}).get('parameters', [])
                    slim_code += ', '.join([f"{p['typeName']['name']} {p['name']}" for p in params])
                    slim_code += ") "
                    if sub_node.get('returnParameters'):
                        return_params = sub_node['returnParameters']['parameters']
                        slim_code += f"returns ({', '.join([r['typeName']['name'] for r in return_params])}) "
                    slim_code += "{\n"
                    for stmt in sub_node.get('body', {}).get('statements', []):
                        slim_code += process_statement(stmt)
                    slim_code += "}\n\n"
    return slim_code
def handle_elements(ast, elements):
    new_elements = set()
    for element in elements:
        function_with_var = None
        for node in ast['children']:
            if node.get('type') == 'ContractDefinition':
                for sub_node in node['subNodes']:
                    if sub_node.get('type') == 'FunctionDefinition':
                        if element == sub_node['name']:
                            new_elements.add(element)
                            break
                        elif any(var.get('name') == element for stmt in sub_node.get('body', {}).get('statements', [])
                                 for var in stmt.get('variables', [])):
                            function_with_var = sub_node['name']

        if function_with_var:
            new_elements.add(function_with_var)
        else:
            new_elements.add(element)

    return list(new_elements)

def read_solidity_code(filename):
    with open(filename, 'r') as file:
        return file.read()

def extract_slim_code_from_original_code(code, elements):
    slim_code = ""
    for element in elements:
        function_pattern = re.compile(r'(function\s+' + element + r'\s*\(.*?\)\s*(\{(?:[^{}]*|\{.*\})*\}))', re.DOTALL)
        match = function_pattern.search(code)
        if match:
            slim_code += match.group(0) + '\n\n'
    return slim_code

def add_contract_wrapper(slim_code, global_vars):
    header = "pragma solidity ^0.8.0;\n\ncontract Addresses {\n\n"
    footer = "\n}\n"
    global_var_declarations = "\n".join(f"    {var}" for var in global_vars)
    return f"{header}{global_var_declarations}\n\n{slim_code}{footer}"

def get_slimcode(sol_code,error_message):
    oringnal_code = read_solidity_code(sol_code)
    code_line = extract_code_from_error_message(error_message)
    elements = remove_solidity_keywords(code_line)
    ast = capture_ast_from_command(sol_code)
    nomoral_json = normalize_json(ast)
    solidity_ast = json.loads(nomoral_json)
    new_elements = handle_elements(solidity_ast, elements)
    slim_code = extract_slim_code_from_original_code(oringnal_code, new_elements)
    global_vars = extract_global_variables(solidity_ast)
    complete_code = add_contract_wrapper(slim_code, global_vars)
    return complete_code
