import ast
import asyncio
import calendar
import copy
from collections import defaultdict
from contextlib import contextmanager
import datetime
import functools
import json
import os
import math
import pickle
import platform
import pprint
import psutil
import pytz
import re
import shutil
import sys
import time
import traceback

import jsonpickle
import numpy as np
from pytz import timezone
from ruamel.yaml.scalarstring import LiteralScalarString
from ruamel.yaml import YAML

from alg_util import is_numpy_type, randomword

EVALPLUS_OBJ = {'base_score': lambda x: x,
    'plus_score': lambda x: x,
    'hybrid_score': lambda x: x,
    'weighted_base_score': lambda x: x,
    'weighted_plus_score': lambda x: x,
    'weighted_hybrid_score': lambda x: x,
    'wall_time_sec': lambda x: -x,
    'user_time_sec': lambda x: -x,
    'sys_time_sec': lambda x: -x,
    'num_instructions': lambda x: -x,
    'memory_usage_mb': lambda x: -x}
SCICODE_OBJ = {'problem_acc': lambda x: x,
    'subproblem_acc': lambda x: x,
    'correct_prob_num': lambda x: x,
    'correct_subprob_num': lambda x: x}
SLEEP_TIME = 5


def extract_evalplus(result_file, logger=None):
    assert os.path.exists(result_file); result_dict = {}
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "(base tests)" in line:
                score = float(lines[i+1].rstrip().rsplit()[1])
                assert 0.0 <= score <= 1.0
                result_dict['base_score'] = score
            if "(base + extra tests)" in line:
                score = float(lines[i+1].rstrip().rsplit()[1])
                assert 0.0 <= score <= 1.0
                result_dict['plus_score'] = score
            # Linux performance metrics
            if platform.system() == 'Linux':
                if "Maximum resident set size (kbytes)" in line:
                    result_dict['memory_usage_mb'] = float(line.split()[-1])/1e3
                if "Elapsed (wall clock) time" in line:
                    result_dict['wall_time_sec'] = time_to_sec(line.split()[-1])
                if "User time" in line:
                    result_dict['user_time_sec'] = float(line.split()[-1])
                if "System time" in line:
                    result_dict['sys_time_sec'] = float(line.split()[-1])
            else: # MacOS performance metrics
                if "peak memory footprint" in line:
                    result_dict['memory_usage_mb'] = float(line.split()[0])/1e6
                if "instructions retired" in line:
                    result_dict['num_instructions'] = float(line.split()[0])
                if "real" in line and "user" in line and "sys" in line:
                    result_dict['wall_time_sec'] = float(line.split()[0])
                    result_dict['user_time_sec'] = float(line.split()[2])
                    result_dict['sys_time_sec'] = float(line.split()[4])

        assert "base_score" in result_dict and "plus_score" in result_dict
        result_dict['hybrid_score'] = \
            0.5 * result_dict["base_score"] + 0.5 * result_dict["plus_score"]
    except:
        stack_trace = traceback.format_exc()
        with open(result_file + ".err", "w") as f: f.write(stack_trace)
        if logger is None:
            print("Evalplus extraction failed: %s" % result_file)
            print(stack_trace)
        else:
            logger.info("Evalplus extraction failed: %s" % result_file)
            logger.info(stack_trace)
    finally:
        return result_dict


def calc_weighted_evalplus_score(result_dir,
    evalplus_weights,
    normalize=True,
    debug_weights=False):
    if isinstance(evalplus_weights, str):
        assert os.path.exists(evalplus_weights)
        with open(evalplus_weights, 'r') as f:
            evalplus_weights = json.load(f)
    else: assert isinstance(evalplus_weights, dict)

    eval_json = os.path.join(result_dir, 'eval_results.json')
    assert os.path.exists(eval_json)
    with open(eval_json, 'r') as f: eval_dict = json.load(f)

    base_score = 0.0; max_base_score = 0.0
    plus_score = 0.0; max_plus_score = 0.0
    for task_id, result in eval_dict['eval'].items():
        base_weight = evalplus_weights['base_weights'][task_id]
        plus_weight = evalplus_weights['plus_weights'][task_id]
        if debug_weights:
            base_weight = 1.0; plus_weight = 1.0

        max_base_score += base_weight; max_plus_score += plus_weight
        if result[0]['base_status'] == "pass": base_score += base_weight
        if result[0]['plus_status'] == "pass": plus_score += plus_weight

    if normalize:
        base_score /= max_base_score; plus_score /= max_plus_score
    return base_score, plus_score


def collect_stats_from_chat(result_dict, *args, **kwargs):
    # pprint.pprint(groupchat_messages, width=120); time.sleep(999999)
    if 'eval_stats' not in result_dict: result_dict['eval_stats'] = {}
    stats_dict = result_dict['eval_stats']

    if 'agent_chat_count' not in stats_dict:
        stats_dict['agent_chat_count'] = {}
    if 'agent_code_count' not in stats_dict:
        stats_dict['agent_code_count'] = {}
    if 'agent_chat_time' not in stats_dict:
        stats_dict['agent_chat_time'] = 0.0

    for message in kwargs.get('groupchat_messages', []):
        agent_name = message['name']
        if not agent_name.endswith("_Expert"):
            continue

        if agent_name not in stats_dict['agent_chat_count']:
            stats_dict['agent_chat_count'][agent_name] = 0
        if agent_name not in stats_dict['agent_code_count']:
            stats_dict['agent_code_count'][agent_name] = 0

        stats_dict['agent_chat_count'][agent_name] += 1
        code = parse_code2(message['content'])
        if code is not None:
            stats_dict['agent_code_count'][agent_name] += len(code)

    stats_dict['agent_chat_time'] += kwargs.get('time_elapsed', 0.0)
    # pprint.pprint(result_dict); time.sleep(999999)


def extract_function_from_code_regex(code_string, function_name):
    pattern = rf'''
        ((?:@\w+\s*\n)*\s*)        # Optional decorators
        (def\s+{re.escape(function_name)}\s*        # Function definition
        \([^)]*\)                  # Parameters (anything inside parentheses)
        \s*(?:->\ *\w+)?           # Optional return type annotation
        \s*:)                      # Colon after function definition
        \s*                        # Possible whitespace after colon
        ((?:\ {4}.*?\n)*           # Function body (indented lines)
        (?:\ {4}.*?)?)?            # Last line might not have newline
    ''', re.VERBOSE | re.MULTILINE | re.DOTALL

    match = re.search(pattern, code_string)

    if match:
        # Return full matched function definition including decorators and body
        return match.group(0).rstrip()

    return None


def extract_function_from_code_tokenize(code_string, function_name):
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code_string).readline))

        # Find the start of the function
        function_start_index = None
        for i, token in enumerate(tokens):
            if (token.type == tokenize.NAME and
                token.string == 'def' and
                tokens[i+1].string == function_name):
                function_start_index = i
                break

        if function_start_index is None:
            return None

        # Collect the function code
        function_lines = []
        current_line = None
        indent_level = None
        in_function = False

        for token in tokens[function_start_index:]:
            if not in_function and token.type == tokenize.NAME and token.string == 'def':
                in_function = True

            if in_function:
                # Track the first line of the function
                if current_line is None:
                    current_line = token.line

                # Track initial indentation
                if indent_level is None and token.type == tokenize.INDENT:
                    indent_level = token.start[1]

                # Collect lines
                if token.type in [tokenize.NEWLINE, tokenize.NL]:
                    function_lines.append(current_line)
                    current_line = None

                # Stop when dedent occurs at the original indentation level
                if token.type == tokenize.DEDENT:
                    break

        # Add the last line if exists
        if current_line:
            function_lines.append(current_line)

        return ''.join(function_lines).rstrip()

    # except tokenize.TokenError:
    except Exception as e:
        return None


def extract_function_from_code(code_string, function_name):
    """
    Extracts and returns the source code of the specified function from a given source code string.

    Uses ast.get_source_segment() to extract the precise source code segment.

    :param code_string: String containing Python source code
    :param function_name: Name of the function to extract
    :return: String containing the source code of the function, including its docstring,
             or None if the function is not found
    """

    if code_string is None:
        return "def %s(): pass\n" % function_name

    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)

        # Iterate through all nodes in the AST
        function_code = None
        for node in ast.walk(tree):
            # Check if the node is a function definition
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                # Use get_source_segment to extract the exact source code
                function_code = ast.get_source_segment(code_string, node)
                # Fall back method if get_source_segment fails
                if function_code is None: function_code = ast.unparse(node)

        # Additional fall back methods using tokenize and regex
        # if function_code is None:
        #     function_code = extract_function_from_code_tokenize(code_string, function_name)
        # if function_code is None:
        #     function_code = extract_function_from_code_regex(code_string, function_name)
        if function_code is None:
            function_code = code_string

        return function_code

    except Exception as e:
        print(f'{function_name} not found with error: {e}')
        return code_string


def extract_comments_from_code(text, incl_single_comments=False):
    """
    Extract comment blocks from Python source code, including docstrings.

    Args:
        text (str): Input text containing Python code and comments

    Returns:
        list: A list of extracted comment blocks
    """

    # try:
    # Regex pattern to match:
    # 1. Text inside triple quotes (docstrings)
    # 2. Single-line comments starting with #, including those after code
    if incl_single_comments:
        comment_pattern = r'(""".*?"""|\'\'\'.*?\'\'\'|(?:^|\s*)#[^\n]*)'
    else:
        comment_pattern = r'(""".*?"""|\'\'\'.*?\'\'\')'

    # Re-use flags for multiline and dot matching
    flags = re.MULTILINE | re.DOTALL

    # Find all comment blocks
    comments = re.findall(comment_pattern, text, flags)

    # Clean up the comments
    processed_comments = []
    for comment in comments:
        # Remove triple quotes from docstrings
        cleaned_comment = re.sub(r'^(\'\'\'|""")|((\'\'\'|""")$)', '', comment).strip()

        # Ensure # comments are preserved
        if cleaned_comment.startswith('#') or not cleaned_comment:
            cleaned_comment = comment.strip()

        if cleaned_comment:
            processed_comments.append(cleaned_comment)

    return "\n".join(processed_comments)
    # except:
    #     return text

def extract_background_from_code(code, index=0):
    """
    Extract contiguous blocks of comments that begin with '# Background'
    from Python code.

    Args:
        code (str): Python source code as a string
        index (int): Index of comment block
    Returns:
        list: Comment block at index
    """
    # Split code into lines and strip whitespace
    lines = code.split('\n')

    # Variables to track comment blocks
    comment_blocks = []
    current_block = []
    in_block = False

    for line in lines:
        stripped = line.strip()

        # Check for start of new background comment block
        if stripped.lower().startswith('# background'):
            # If we were already in a block, save it first
            if in_block and current_block:
                comment_blocks.append('\n'.join(current_block))
                current_block = []

            in_block = True
            current_block.append(stripped)

        # Continue existing comment block
        elif in_block and stripped.startswith('#'):
            current_block.append(stripped)

        # End of comment block
        elif in_block:
            comment_blocks.append('\n'.join(current_block))
            current_block = []
            in_block = False

    # Add final block if code ends with a comment block
    if in_block and current_block:
        comment_blocks.append('\n'.join(current_block))

    try:
        return comment_blocks[index]
    except:
        return None


def extract_name_from_function(code, raise_error=True):
    """
    Extract the function or class name from a given Python code string using AST.

    Args:
        code (str): String containing Python code

    Returns:
        Optional[str]: Name of the function or class if found, None otherwise
    """
    # Parse the code into an AST
    tree = ast.parse(code)

    # Look for the first function or class definition
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return node.name


def load_imports_from_string(import_string):
    """
    Load import statements and add modules to namespace.
    Handles multiple import formats:
    - import module
    - import module as alias
    - from module import submodule
    - from module import submodule as alias
    - Multiple imports on one line (from module import x, y, z)
    """

    # Start with a copy of the global namespace
    namespace = globals().copy()
    # Split into individual import statements
    import_lines = [line.strip() for line in import_string.strip().split('\n')
                   if line.strip() and not line.startswith('#')]

    for line in import_lines:
        # try:
        if line.startswith('from'):
            # Handle "from module import submodule" style imports
            parts = line.split()
            module_path = parts[1]
            # Everything after 'import' is part of targets
            targets = ' '.join(parts[3:]).split('#')[0].strip()  # Remove inline comments

            # Import the base module
            __import__(module_path)
            module = sys.modules[module_path]

            # Handle multiple targets (e.g., from math import sin, cos, tan)
            for target in targets.split(','):
                target = target.strip()
                if ' as ' in target:
                    name, alias = [x.strip() for x in target.split(' as ')]
                    namespace[alias] = getattr(module, name)
                else:
                    namespace[target] = getattr(module, target)
        else:
            # Handle "import module" style imports
            parts = line.split('#')[0].strip().split()  # Remove inline comments
            if 'as' in parts:
                # Handle "import module as alias"
                as_index = parts.index('as')
                module_name = parts[1]
                alias = parts[as_index + 1]
                module = __import__(module_name)
                namespace[alias] = module
            else:
                # Handle "import module"
                module_name = parts[1]
                module = __import__(module_name)
                namespace[module_name] = module
        # except Exception as e:
        #     traceback.print_exc()
        #     print(f"Error importing {line}: {str(e)}")

    return namespace


def eval_function_from_string(namespace, func_string, func_name, compile=False):
    """
    Creates a function from a string definition and adds it to the given namespace.

    Args:
        namespace (dict): The namespace where the function will be created
        func_string (str): String containing the function definition
        func_name (str): Expected name of the function
        compile (bool): Whether to pre-compile the code to catch syntax errors early

    Returns:
        callable: The created function object
    """

    # Pre-compile the function code to catch syntax errors early
    if compile: func_string = compile(func_string, '<string>', 'exec')
    # Execute function definition in namespace
    exec(func_string, namespace)
    # Verify function exists in namespace
    if func_name not in namespace:
        raise RuntimeError(f"Function '{func_name}' not found in namespace after execution")
    return namespace[func_name]
    # except Exception as e:
    #     traceback.print_exc()
    #     raise RuntimeError(f"Error creating function: {str(e)}")


def extract_elements_from_code(content):
    """
    Extracts and organizes Python code elements (imports, classes, and functions)
    from a string of source code.

    Args:
        content (str): String containing Python source code

    Returns:
        str | None: A formatted string containing the extracted code elements
        organized by type, or None if parsing fails
    """

    # try:
    # Parse the Python code into an AST
    tree = ast.parse(content)
    # Extract imports, functions, and classes
    imports = []; classes = []; functions = []

    for node in ast.walk(tree):
        # Get import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(content, node))
        # Get class definitions
        elif isinstance(node, ast.ClassDef):
            classes.append(ast.get_source_segment(content, node))
        # Get function definitions
        elif isinstance(node, ast.FunctionDef):
            functions.append(ast.get_source_segment(content, node))

    # Combine all extracted elements
    extracted_code = ""
    if len(imports) > 0:
        extracted_code += "\n".join(imports) + "\n\n"
    if len(classes) > 0:
        extracted_code += "\n\n".join(classes) + "\n\n"
    if len(functions) > 0:
        extracted_code += "\n\n".join(functions) + "\n"
    return extracted_code

    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    #     return None


def extract_code_from_chat(chat_result):
    code = ""
    result = parse_code2(chat_result.summary)
    if result is not None:
        code = result
    else:
        for msg_dict in chat_result.chat_history[::-1]:
            result = parse_code2(msg_dict['content'])
            if result is not None:
                code = result
    return code


def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text


def parse_code2(text):
    """
    Extracts content from markdown code blocks.

    Args:
        text (str): The markdown text containing code blocks

    Returns:
        list: List of tuples (language, content) where language might be None
              if not specified in the code block

    Note:
        - Preserves internal whitespace
        - Handles escaped backticks within code
        - Skips invalid/incomplete code blocks
    """
    pattern = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
    matches = re.finditer(pattern, text, re.DOTALL)

    # results = []
    for match in matches:
        try:
            language = match.group(1)
            # Remove only the trailing newline if it exists, preserve other whitespace
            content = match.group(2).rstrip('\r\n')

            # Skip empty code blocks
            if not content.strip(): continue
            if language.lower() == "python": return content
            # results.append((language, content))

        except (IndexError, AttributeError):
            # Skip malformed matches
            continue

    # return results
    return None


def parse_prompt_template(rsp):
    pattern = r"PROMPT_TEMPLATE: str = '''(.*)'''"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    code_text = code_text.lstrip().rstrip()
    return code_text


def format_prompt(prompt, instruction):
    try:
        prompt = prompt.format(instruction=instruction)
    except:
        try: # If {instruction} not found, search for first braces
            special_word = prompt[prompt.find("{"):prompt.find("}")+1]
            prompt = prompt.replace(special_word, instruction)
        except: # Last resort, just use problem directly
            prompt = instruction
    return prompt


def convert_to_comments(text: str) -> str:
    """Convert a multiline string into Python comments using # symbols."""

    # Type checking
    if not isinstance(text, str): raise TypeError("Input must be a string")
    # Handle empty or whitespace-only input
    if not text.strip(): return ""

    lines = text.strip().split('\n'); new_lines = []
    for line in lines:
        if not line.strip(): new_lines.append("#"); continue
        if line.startswith("#"): new_lines.append(line)
        else: new_lines.append(f"# {line}")

    return '\n'.join(new_lines)


def killtree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        print("Killing child: %s" % child)
        child.kill()

    if including_parent:
        parent.kill()


def flatten(xss):
    return [x for xs in xss for x in xs]


def delete_contents_in_directory(directory_path, verbose=False):
    with os.scandir(directory_path) as entries:
        for entry in entries:
            if verbose: print("Deleting %s" % entry.path)
            try:
                if entry.is_file(): os.unlink(entry.path)
                else: shutil.rmtree(entry.path)
            except:
                if verbose: print("Deletion error\n%s" % traceback.format_exc())


def clear_autogen_cache():
    os.system("rm -rf .cache >/dev/null 2>&1")
    delete_contents_in_directory("/tmp/")
    # os.system("rm -rf /tmp/* >/dev/null 2>&1")


def time_to_sec(time_str):
    """Get seconds from time."""
    fields = [float(x) for x in time_str.split(':')]
    if len(fields) == 2:
        m, s = fields; h = 0.0
    else:
        h, m, s = fields
    return h * 3600.0 + m * 60.0 + s


def unzip(x):
    return [list(x) for x in zip(*x)]


def get_time(date=True, space=True):
    '''Creates a nicely formated timestamp'''
    if date:
        date_str = "%Y-%m-%d %H:%M:%S"
    else:
        date_str = "%H:%M:%S"

    if not space:
        date_str = date_str.replace(":", "-").replace(" ", "_")

    return datetime.datetime.now(timezone('US/Pacific')).strftime(date_str)


def datetime_to_epoch(datetime_str, space=True):
    if space:
        date, time = datetime_str.split()
        h, _m, s = time.split(":")
    else:
        date, time = datetime_str.split("_")
        h, _m, s = time.split("-")
    y, m, d = date.split("-")
    t = datetime.datetime(int(y), int(m), int(d), int(h), int(_m), int(s))
    # tz = pytz.timezone('America/Los_Angeles'); t = t.astimezone(tz)
    return calendar.timegm(t.timetuple())


def sanitize_result_dict(result_dict):
    '''Converts numpy types to python built ins'''
    if isinstance(result_dict, (list, tuple)):
        return [sanitize_result_dict(x) for x in result_dict]
    elif isinstance(result_dict, dict):
        new_result_dict = {}
        for k in result_dict:
            new_result_dict[k] = sanitize_result_dict(result_dict[k])
        return new_result_dict
    elif is_numpy_type(result_dict):
        if isinstance(result_dict, np.ndarray):
            return result_dict.tolist()
        else:
            return result_dict.item()
    else:
        return result_dict


def get_result_from_cache(key, cache_dir='/tmp/'):
    try:
        cache_file = os.path.join(cache_dir, str(key) + '.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                result_dict = pickle.load(f)
            return result_dict
        else:
            return None
    except:
        traceback.print_exc()
        return None


def save_result_to_cache(key, result_dict, cache_dir='/tmp/'):
    tmp_file = os.path.join(cache_dir, randomword(24))
    cache_file = os.path.join(cache_dir, str(key) + '.pkl')
    assert not os.path.exists(cache_file)

    with open(tmp_file, 'wb') as f:
        pickle.dump(result_dict, f)
        f.flush()
        os.fsync(f.fileno())
        f.close()

    os.rename(tmp_file, cache_file)
    assert os.path.exists(cache_file)


def save_global_config(filepath, global_dict):
    def strip_python_tags(s):
        result = []
        for line in s.splitlines():
            idx = line.find("!!python/")
            if idx > -1:
                line = line[:idx]
            result.append(line)
        return '\n'.join(result)

    config_dict = {}
    for key, value in global_dict.items():
        # print(key, value)
        if key.isupper() and not key.startswith('__'):
            config_dict[key] = copy.deepcopy(value)

    # jsonpickle.set_encoder_options('simplejson',
    #     use_decimal=True, sort_keys=True)
    # jsonpickle.set_preferred_backend('simplejson')
    # s = jsonpickle.encode(config_dict, indent=4)
    with open(filepath, 'w') as f:
        # f.write(s)
        YAML().dump(config_dict, f)
    print("Dumped global config to file: %s" % filepath)


def get_indv_config(experiment_dir, config_name="config.yaml"):
    try:
        with open(os.path.join(experiment_dir, config_name), "r") as f:
            exp_cfg = YAML().load(f)
        indv_config = exp_cfg['role_ga_config']['indv_config']
        print("Indv config:"); pprint.pprint(indv_config)
    except:
        traceback.print_exc()
        print("Cannot load indv config!"); time.sleep(3); indv_config = {}
    return indv_config


def get_eval_config(experiment_dir, config_name="config.yaml"):
    try:
        with open(os.path.join(experiment_dir, config_name), "r") as f:
            exp_cfg = YAML().load(f)
        eval_config = exp_cfg['llm_evaluator_config']
        print("Evaluator config:"); pprint.pprint(eval_config)
    except:
        traceback.print_exc()
        print("Cannot load evaluator config!"); time.sleep(3); eval_config = {}
    return eval_config


def yaml_dump(data, output_file, width=80, mode='w'):
    """Dumps data in a human-readable format"""
    def should_use_flow_style(node):
        """Determine if a node should use flow style based on its content"""
        if isinstance(node, list):
            # Use flow style for simple lists (strings, numbers, etc)
            if all(isinstance(item, (str, int, float, bool)) for item in node):
                return True
            return False
        return None  # Let ruamel.yaml decide for other types

    # Format long strings using literal block style
    def format_content(item):
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str) and \
                    ('\n' in value or len(value) > width - 20):
                    item[key] = LiteralScalarString(value)
                elif isinstance(value, (dict, list)):
                    format_content(value)
        elif isinstance(item, list):
            for i, value in enumerate(item):
                if isinstance(value, dict):
                    format_content(value)

    # Initialize YAML with round-trip capabilities
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = width  # Set line width for wrapping
    yaml.default_flow_style = None
    yaml.allow_unicode = True
    # yaml.indent(mapping=2, sequence=4, offset=2)

    if isinstance(data, str):
        assert os.path.exists(data)
        with open(data, 'r') as f: data = dict(yaml.load(f))

    # Apply formatting
    format_content(data)

    # Write the formatted YAML
    if not output_file.endswith(".yaml"): output_file += ".yaml"
    with open(output_file, mode) as f: yaml.dump(data, f)


class OutputRedirector:
    """
    Class to redirect both stdout and stderr to terminal and a file.
    """
    def __init__(self, file_path, file_ext='.log'):
        """
        Initialize the output redirector.

        :param file_path: Path to the file where output will be written
        """
        # Save the original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.file_path = os.path.splitext(file_path)[0] + file_ext
        self.file = None

        # Custom stream class
        class TeeStream:
            def __init__(self, original_stream, file_stream):
                self.original_stream = original_stream
                self.file_stream = file_stream

            def write(self, data):
                # Write to original stream (terminal)
                self.original_stream.write(data)
                # Write to file
                self.file_stream.write(data)

            def flush(self):
                # Ensure both streams are flushed
                self.original_stream.flush()
                self.file_stream.flush()

            # Implement additional methods to mimic stream behavior
            def isatty(self):
                return False

        self.TeeStream = TeeStream

    def enable(self):
        """
        Enable output redirection to file and terminal.
        """
        # Open the file in write mode (overwriting previous content)
        self.file = open(self.file_path, 'w')

        # Create and set custom stdout and stderr streams
        sys.stdout = self.TeeStream(self.original_stdout, self.file)
        sys.stderr = self.TeeStream(self.original_stderr, self.file)

    def disable(self):
        """
        Disable output redirection and restore original streams.
        """
        # Restore original stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Close the file if it's open
        if self.file:
            self.file.close()
            self.file = None


if __name__ == "__main__":
    imports = "import numpy as np"
    namespace = load_imports_from_string(imports)
    test_func = \
"""
# Background
# blah blah blah
# blah blah blah
#
# blah blah blah

# Background
# asdf
# adsf asdf

"""
    print(convert_to_comments(""))
    # print(extract_background_from_code(test_func, 0))
    # from scicode.parse.parse import extract_function_name
    # # test_func_name = extract_function_name(test_func)
    # test_func_name = extract_name_from_function(test_func)
    # print(test_func_name)
    # print(eval_function_from_string(namespace, test_func, test_func_name))
    # print(parse_comment_block(x))
    # yaml_dump(sys.argv[1], sys.argv[1].replace(".yaml", ".p.yaml"))
