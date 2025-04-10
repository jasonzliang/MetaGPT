import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import time

import numpy as np

from scicode.gen.models import get_model_function
from scicode.parse.parse import extract_function_name, get_function_from_code, \
    read_from_jsonl

from util import extract_function_from_code, extract_name_from_function
from util import extract_background_from_code, convert_to_comments
# from scicode.parse.parse import H5PY_FILE

PROB_NUM = 65
DEV_PROB_NUM = 15
STEP_NUM = 288
DEV_STEP_NUM = 50
ALL_PROB_NUM = PROB_NUM + DEV_PROB_NUM
ALL_STEP_NUM = STEP_NUM + DEV_STEP_NUM

DEFAULT_PROMPT_TEMPLATE = Path("scicode_data", "background_comment_template.txt").read_text()
BACKGOUND_PROMPT_TEMPLATE = Path("scicode_data", "multistep_template.txt").read_text()
# DEFAULT_PROMPT_TEMPLATE = Path("scicode_data", "background_comment_template_v2.txt").read_text()
# BACKGOUND_PROMPT_TEMPLATE = Path("scicode_data", "multistep_template_v2.txt").read_text()

H5PY_FILE = os.path.join("scicode_data/test_data.h5")
CLEANUP_TMP_FILES = False
TEST_TIMEOUT = 300


def extract_python_script(response: str):
    # We will extract the python script from the response
    if '```' in response:
        python_script = response.split("```python")[1].split("```")[0] if '```python' in response else response.split('```')[1].split('```')[0]
    else:
        print("scicode_eval.extract_python_script error, code already extracted?")
        # print(response + "\n")
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script


class Gencode:
    def __init__(self, output_dir: Path,
                 prompt_dir: Path,
                 with_background: bool,
                 include_bg_comments: bool = True,
                 model: str = None,
                 temperature: float = None,
                 llm_eval_func: callable = None):
        self.model = model
        self.temperature = temperature
        self.llm_eval_func = llm_eval_func

        self.output_dir = output_dir
        self.prompt_dir = prompt_dir
        self.with_background = with_background
        self.include_bg_comments = include_bg_comments or with_background
        self.previous_llm_code = []

    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"

    def _get_output_file_path(self, prob_id, num_steps):
        res_output_dir = Path(self.output_dir,
            Path(self.model).parts[-1],
            self._get_background_dir())
        res_output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = Path(res_output_dir, f"{prob_id}.{num_steps}.py")
        return output_file_path

    def _save_prompt_with_steps(self, prob_data: dict, prompt: str, num_steps: int) -> None:
        output_dir = Path(self.prompt_dir,
            Path(self.model).parts[-1],
            self._get_background_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{prob_data['problem_id']}.{num_steps}.txt"
        output_file_path.write_text(prompt, encoding="utf-8")

    def _save_response_with_steps(self,
        prob_data: dict,
        background: str,
        func_code: str,
        previous_code: str,
        num_steps: int) -> None:

        prob_id = prob_data["problem_id"]
        # python_code = extract_python_script(response)
        output_file_path = self._get_output_file_path(prob_id, num_steps)
        if background is not None:
            output = f'{previous_code}\n{background}\n\n{func_code}'
        else:
            output = f'{previous_code}\n{func_code}'
        output_file_path.write_text(output, encoding="utf-8")

        return output_file_path

    @staticmethod
    def _process_problem_code(prob_data: dict, num_steps: int) -> str:
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string

    def _process_problem_steps(self, problem_data: dict, num_steps: int):
        """Process problem data and return previous steps and next steps"""
        def _get_background(i, include_desc=True):
            return_list = []
            description = problem_data["sub_steps"][i]["step_description_prompt"]
            if include_desc: return_list.append(description)

            background = None
            if self.with_background:
                background = problem_data["sub_steps"][i]["step_background"]
            elif self.previous_llm_code[i] is not None:
                background = self.previous_llm_code[i]['background']

            if self.include_bg_comments and background is not None:
                return_list.append(convert_to_comments(background))
            return "\n\n".join(return_list)

        output_lines = []; next_step = []; previous_code = []
        for i in range(num_steps - 1):
            output_lines.append(_get_background(i))
            output_lines.append(self.previous_llm_code[i]['code'])
            output_lines.append("------")

            previous_code.append(_get_background(i, include_desc=False))
            previous_code.append(self.previous_llm_code[i]['code'])

        next_step.append(_get_background(num_steps - 1))
        next_step.append(self._process_problem_code(problem_data, num_steps))

        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n\n".join(previous_code)

        return output_str, next_step_str, previous_code_str

    def _generate_prompt_with_steps(self, prob_data: dict, num_steps: int,
                                   prompt_template=DEFAULT_PROMPT_TEMPLATE):
        # parse the input file and extract the content
        problem_steps_str, next_step_str, previous_code_str = \
            self._process_problem_steps(prob_data, num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n{previous_code_str}\n'

    def generate_response_with_steps(self,
        prob_data: dict,
        num_steps: int,
        tot_steps: int,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        result_dict: dict = None,
        save: bool = True) -> None:
        """

        Args:
            prob_data (dict): Dict of the problem
            num_steps (int): Current generating step
            tot_steps (int): Total step of the problem
            prompt_template (str): template for prompt
            result_dict (dict): Evaluation dict
            save (bool, optional): Save prompt and model response. Defaults to True.
        """
        prob_id = prob_data["problem_id"]
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    # 291 - 3 = 288 actual subproblems (3 are provided)
                    if (prob_id == "13" and prev_step == 5) or (prob_id == "62" and prev_step == 0)\
                            or (prob_id == "76" and prev_step == 2):
                        prev_file_path = Path("scicode_data", f"{prob_id}.{prev_step+1}.txt")
                    else:
                        prev_file_path = self._get_output_file_path(prob_id, prev_step + 1)

                    if prev_file_path.is_file() and prev_file_path.stat().st_size > 0:
                        prev_file_content = prev_file_path.read_text(encoding='utf-8')
                        assert prev_file_content is not None
                        func_header = prob_data["sub_steps"][prev_step]["function_header"]
                        func_name = extract_name_from_function(func_header)
                        func_code = extract_function_from_code(prev_file_content, func_name)
                        background = extract_background_from_code(prev_file_content, index=-1)
                        assert func_name is not None and func_code is not None

                        self.previous_llm_code[prev_step] = {
                            'code': func_code,
                            'name': func_name,
                            'header': func_header,
                            'background': background
                        }
                    else:
                        try:
                            self.generate_response_with_steps(
                                prob_data,
                                prev_step,
                                tot_steps,
                                prompt_template,
                                result_dict,
                                save)
                        except:
                            raise Exception(f'Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}.')

        prompt, previous_code = self._generate_prompt_with_steps(
            prob_data, num_steps, prompt_template)
        if save: self._save_prompt_with_steps(prob_data, prompt, num_steps)

        model_kwargs = {}
        if "claude" in self.model: model_kwargs["max_tokens"] = 4096
        model_kwargs["temperature"] = self.temperature

        # write the response to a file if it doesn't exist
        output_file_path = self._get_output_file_path(prob_id, num_steps)
        if output_file_path.is_file() and output_file_path.stat().st_size > 0:
            print("Output file exists, skipping: %s" % output_file_path)
            return None

        if self.llm_eval_func is None:
            model_fct = get_model_function(model, **model_kwargs)
            response_from_llm = model_fct(prompt)
        else:
            result_dict['code_library'] = [self.previous_llm_code[i] for i in range(num_steps - 1) if self.previous_llm_code[i] is not None]
            result_dict['imports'] = prob_data['required_dependencies']
            response_from_llm = self.llm_eval_func(
                f"{prob_id}.{num_steps}", prompt, result_dict)

        func_header = prob_data["sub_steps"][num_steps - 1]["function_header"]
        func_name = extract_name_from_function(func_header)
        python_script = extract_python_script(response_from_llm)
        func_code = extract_function_from_code(python_script, func_name)
        background = extract_background_from_code(python_script, index=0)
        assert func_name is not None and func_code is not None

        self.previous_llm_code[num_steps - 1] = {
            'code': func_code,
            'name': func_name,
            'header': func_header,
            'background': background
        }
        return self._save_response_with_steps(
            prob_data,
            background,
            func_code,
            previous_code,
            num_steps)


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model name"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results", "generated_code"),
        help="Output directory",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("eval", "data", "problems_all.jsonl"),
        help="Input directory",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("eval_results", "prompt"),
        help="Prompt directory",
    )
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Include problem background if enabled",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Generation temperature",
    )
    return parser


def main(model: str,
         output_dir: Path,
         input_path: Path,
         prompt_dir: Path,
         with_background: bool,
         temperature: float
) -> None:
    gcode = Gencode(
        model=model, output_dir=output_dir,
        prompt_dir=prompt_dir,  with_background=with_background, temperature=temperature
    )
    prompt_template = BACKGOUND_PROMPT_TEMPLATE if with_background else DEFAULT_PROMPT_TEMPLATE
    data = read_from_jsonl(input_path)
    for problem in data:
        prob_id = problem['problem_id']
        steps = len(problem['sub_steps'])
        print(f'Generating {prob_id}...')
        for i in range(steps):
            if (prob_id == "13" and i == 5) or (prob_id == "62" and i == 0)\
                    or (prob_id == "76" and i == 2):
                continue
            gcode.generate_response_with_steps(problem, i + 1, steps, model, prompt_template)

############## TEST CODE ##############

def _get_background_dir(with_background):
    return "with_background" if with_background else "without_background"


def _run_script(script_path): # Original run_script
    try:
        subprocess.run(['python', script_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_path}: {e}")
        print(e.output)
        return 1
    except subprocess.TimeoutExpired as e:
        print(f"Runtime error while running script {script_path}: {e}")
        return 2


def _run_script2(script_path): # New version that returns output text
    """
    Run a Python script and return both its exit status and output.

    Args:
        script_path (str): Path to the Python script to run

    Returns:
        tuple: (exit_code, output_text)
            - exit_code: 0 for success, 1 for script error, 2 for timeout
            - output_text: String containing stdout/stderr from the script
    """
    try:
        result = subprocess.run(
            ['python', script_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT
        )
        # Return success, error (0, 1) and the output
        return result.returncode, result.stdout + result.stderr

    except subprocess.TimeoutExpired as e:
        # Handle timeout case
        output = f"Timeout ({TEST_TIMEOUT}s) while running script {script_path}: {e}\n"
        if e.stdout:
            output += f"Stdout: {e.stdout}\n"
        if e.stderr:
            output += f"Stderr: {e.stderr}\n"
        print(output); return 2, output


def test_code(model_name, code_dir, log_dir, output_dir,
              jsonl_path, dev_set=False, with_background=False):

    assert os.path.exists(H5PY_FILE)
    jsonl_data = read_from_jsonl(jsonl_path)
    json_dct = {}
    json_idx = {}

    for prob_data in jsonl_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = jsonl_data.index(prob_data)
    start_time = time.time()

    code_dir_ = Path(code_dir, model_name, _get_background_dir(with_background))
    if not code_dir_.exists():
        print("scicode_eval.test_code error, code dir does not exist: %s" % code_dir_)
        return {'problem_acc': 0.0,
            'subproblem_acc': 0.0,
            'correct_prob_num': 0,
            'correct_subprob_num': 0,
            'correct_dict': {}}

    tmp_dir = Path('/tmp', f'tmp_{start_time}')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for file_path in code_dir_.iterdir():
        # Add additional check if file is .py extension
        if file_path.is_file() and file_path.suffix.lower() == ".py":
            file_name = file_path.stem
            file_id = file_name.split(".")[0]
            file_step = file_name.split(".")[1]

            code_content = file_path.read_text(encoding='utf-8')
            json_content = jsonl_data[json_idx[file_id]]
            step_id = json_content["sub_steps"][int(file_step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(file_step) - 1]["test_cases"]
            assert_file = Path(tmp_dir, f'{step_id}.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(f"""

from scicode.parse.parse import process_hdf5_to_tuple

""")
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)}, '{H5PY_FILE}')" + '\n')
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split('\n'):
                        f.write(line + '\n')

    correct_prob = np.zeros(ALL_PROB_NUM)
    tot_prob = np.zeros(ALL_PROB_NUM)
    correct_step = []
    correct_dict = {}

    for i in range(ALL_PROB_NUM):
        correct_dict[f'{i+1}'] = []

    sorted_iterdir = sorted(tmp_dir.iterdir())
    for file_path in sorted_iterdir:
        if file_path.is_file():
            func_id = file_path.stem
            prob_id = func_id.split('.')[0]
            print(f'Testing function {func_id} ...')
            tot_prob[int(prob_id) - 1] += 1
            logs_dir_ = Path(log_dir, model_name, _get_background_dir(with_background))
            logs_dir_.mkdir(parents=True, exist_ok=True)
            logs_file = Path(logs_dir_, f'{file_path.stem}.txt')
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    content = f.read().splitlines()
                    if content[0] == 'pass':
                        correct_prob[int(prob_id) - 1] += 1
                        correct_step.append(func_id)
                        correct_dict[prob_id].append(func_id)
                continue

            ret, script_output = _run_script2(file_path)
            if ret == 0:
                correct_prob[int(prob_id) - 1] += 1
                correct_step.append(func_id)
                correct_dict[str(prob_id)].append(func_id)
                with open(logs_file, 'w') as f:
                    f.write('pass')
            elif ret == 1:
                with open(logs_file, 'w') as f:
                    f.write('fail')
            else:
                with open(logs_file, 'w') as f:
                    f.write('time out')

            script_out_file = Path(logs_dir_, f'{file_path.stem}_output.txt')
            with open(script_out_file, 'w') as f: f.write(script_output)
            script_file = Path(logs_dir_, f'{file_path.stem}_test.py')
            shutil.copy(file_path, script_file)

    test_time = time.time() - start_time

    correct_prob_num = sum(1 for i in range(ALL_PROB_NUM) if
                           correct_prob[i] == tot_prob[i]
                           and tot_prob[i] != 0)
    correct_step_num = len(correct_step)
    total_prob = DEV_PROB_NUM if dev_set else PROB_NUM
    total_step = DEV_STEP_NUM if dev_set else STEP_NUM

    print(f'Correct problems: {correct_prob_num}/{total_prob}')
    print(f'Correct steps: {correct_step_num}/{total_step}')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{output_dir}/{model_name}_{_get_background_dir(with_background)}.txt', 'w') as f:
        f.write(f'Correct problems: {correct_prob_num}/{total_prob}\n')
        f.write(f'Correct steps: {correct_step_num}/{total_step}\n\n')
        f.write(f'Correct problems (%): {round(100.0*correct_prob_num/total_prob, 1)}\n')
        f.write(f'Correct steps (%): {round(100.0*correct_step_num/total_step, 1)}\n\n')
        f.write(f'Duration: {test_time} seconds\n')
        f.write('\nCorrect problems: ')
        f.write(f'\n\n{[i + 1 for i in range(ALL_PROB_NUM) if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0]}\n')

    with open(f'{output_dir}/{model_name}_{_get_background_dir(with_background)}.json', 'w', encoding='utf-8') as f:
        json.dump(correct_dict, f, indent=4)

    if CLEANUP_TMP_FILES: shutil.rmtree(tmp_dir)

    total_prob_num = DEV_PROB_NUM if dev_set else PROB_NUM
    total_step_num = DEV_STEP_NUM if dev_set else STEP_NUM
    problem_acc = float(correct_prob_num)/float(total_prob_num)
    subproblem_acc = float(len(correct_step))/float(total_step_num)

    return {'problem_acc': problem_acc,
        'subproblem_acc': subproblem_acc,
        'correct_prob_num': correct_prob_num,
        'correct_subprob_num': len(correct_step),
        'correct_dict': correct_dict}


def test_get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model name"
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path("eval_results", "generated_code"),
        help="Code directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Log directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Eval results directory",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("eval", "data", "problems_all.jsonl"),
        help="Path to jsonl file",
    )
    parser.add_argument(
        "--dev-set",
        action='store_true',
        help="Test dev set if enabled",
    ),
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Include problem background if enabled",
    )
    return parser


def test_main(model: str,
         code_dir: Path,
         log_dir: Path,
         output_dir: Path,
         jsonl_path: Path,
         dev_set: bool,
         with_background: bool
) -> None:
    if not Path(H5PY_FILE).exists():
        raise FileNotFoundError("Please download the numeric test results before testing generated code.")
    model = Path(model).parts[-1]
    test_code(model, code_dir, log_dir, output_dir, jsonl_path, dev_set, with_background)


if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
