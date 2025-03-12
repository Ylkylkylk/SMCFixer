from retrieve_and_repair import get_top1,get_top3,get_top5
import argparse
import json
import torch
import subprocess
import re
from code_scling import get_slimcode

# Command line parameter settings
def parse_arguments():
    parser = argparse.ArgumentParser(description='Solidity Code Assistant')
    parser.add_argument('--file', type=str, help='Path to the Solidity .sol file', required=True)
    parser.add_argument('--top1', action='store_true', help='Use start_test3 function')
    parser.add_argument('--top3', action='store_true', help='Use start_test4 function')
    parser.add_argument('--top5', action='store_true', help='Use start_test5 function')
    return parser.parse_args()

# loading Knowledge-Base
def load_embeddings_from_json(file_path):
    with open(file_path, 'r') as file:
        embeddings_list = json.load(file)
    tensor_embeddings = [torch.tensor(embedding) for embedding in embeddings_list]
    return tensor_embeddings

# compiling solidity code
def compile_solidity(source_file):
    try:
        result = subprocess.run(['solcjs', '--bin', '--abi', source_file], capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

#filtering error message
def clean_error_message(error_message):
    clean_message = error_message.replace("Error: ", "")
    clean_message = re.sub(r'\n', ' ', clean_message)
    clean_message = re.sub(r'--> .*?:\d+:\d+:', '', clean_message)
    return clean_message.strip()

def read_solidity_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_solidity_code(text):
    match = re.search(r'```solidity\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def main():
    args = parse_arguments()
    solidity_file = args.file
    context_embeddings = load_embeddings_from_json('./Knowledge_Base_embedding.json')
    solidity_code = read_solidity_file(solidity_file)
    success, output_or_error = compile_solidity(solidity_file)

    attempts = 0
    max_attempts = 5

    if args.top1:
        run_function = get_top1()
    elif args.top3:
        run_function = get_top3()
    elif args.top5:
        run_function = get_top5()

    while not success and attempts < max_attempts:
        print(f"Compile failed, attempt {attempts + 1} processing error...")
        cleaned_error_message = clean_error_message(output_or_error)
        slim_code = get_slimcode(solidity_file,output_or_error)
        modified_code = run_function(cleaned_error_message, context_embeddings, slim_code)
        clean_modified_code = extract_solidity_code(modified_code)

        temp_file_path = 'test.sol'
        with open(temp_file_path, 'w', encoding='utf-8') as file:
            file.write(clean_modified_code)

        success, output_or_error = compile_solidity(temp_file_path)
        if success:
            print("Compile success")
            print(output_or_error)
        attempts += 1

    if not success:
        print("Failed to compile after 5 attempts.")

if __name__ == "__main__":
    main()
