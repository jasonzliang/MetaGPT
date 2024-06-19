import copy
import fire
import os
import pprint
import re
import sys
import time

import asyncio
from agent_protocol.models import StepRequestBody
from agent_protocol_client import (
    Configuration,
    ApiClient,
    StepRequestBody,
    TaskRequestBody,
    AgentApi,
)

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import write_jsonl

from util import get_time


async def autogpt(prompt):
    # Defining the host is optional and defaults to http://localhost
    # See configuration.py for a list of all supported configuration parameters.
    configuration = Configuration(host="http://localhost:8000")
    # Enter a context with an instance of the API client
    async with ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = AgentApi(api_client)
        task_request_body = TaskRequestBody(input="Complete the following python code and write to result.txt, while ensuring correctness and accuracy:\n%s" % prompt)
        # task_request_body = TaskRequestBody(input="Complete or implement the following Python code and write the output to result.txt. Write correct, efficient, correct code with NO other texts. Python code:\n%s" % prompt)

        response = await api_instance.create_agent_task(
            task_request_body=task_request_body
        )
        print("The response of AgentApi->create_agent_task:\n")
        print(response)
        print("\n\n")

        task_id = response.task_id; i = ""

        while (
            step := await api_instance.execute_agent_task_step(
                task_id=task_id, step_request_body=StepRequestBody(input=i)
            )
        ) and step.is_last is False:
            print("The response of AgentApi->execute_agent_task_step:\n")
            print(step)
            print("\n\n")
            # i = str(int(i) + 1)

        print("Agent finished its work!")

        artifacts = await api_instance.list_agent_task_artifacts(task_id=task_id)
        for artifact in artifacts:
            try:
                if type(artifact) is tuple:
                    artifact = artifact[1]
                if type(artifact) is list:
                    artifact = artifact[0]
                if artifact.file_name == "result.txt":
                    content = await api_instance.download_agent_task_artifact(
                        task_id=task_id, artifact_id=artifact.artifact_id
                    )
                    print(f'The content of the result.txt is {content})')
                    return content.decode()
            except:
                continue

        print("The agent did not create the result.txt file.")
        return ""


def generate_code_prompt(example: dict) -> str:
    return example['instruction']


async def eval_humaneval(
    result_dir="results/humaneval_results_%s" % get_time(space=False),
    # result_dir="results/humaneval_results_1718217224",
):
    problems = get_human_eval_plus()
    eval_name = "humaneval"
    results = []

    for task_id, problem in problems.items():
        task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
        os.makedirs(task_id_dir, exist_ok=True)
        result_file = os.path.join(task_id_dir, "0.py")
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            continue

        sample = {"instruction": problem['prompt'],
            "input": problem['base_input']}
        prompt = generate_code_prompt(sample)
        print("\n\n#### Task ID: %s, Prompt:\n%s" % (task_id, prompt))

        output = await autogpt(prompt)
        with open(result_file, 'w') as f:
            f.write(output)
        results.append({'task_id': task_id, 'solution': output})

    os.system("evalplus.sanitize --samples %s >/dev/null" % result_dir)
    os.system("rsync -avz %s-sanitized/ %s >/dev/null" % \
        (result_dir, result_dir))
    os.system("rm -rf %s-sanitized" % result_dir)
    os.system("evalplus.evaluate --dataset %s --samples %s | tee %s"
        % (eval_name, result_dir, os.path.join(result_dir, "evalplus.txt")))
    os.system("cp %s %s" % (__file__, result_dir))


if __name__ == "__main__":
    fire.Fire(eval_humaneval)
