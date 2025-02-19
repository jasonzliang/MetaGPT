import copy
import hashlib
import importlib
import json
import logging
import os
import pprint
import random
import re
import socket
import subprocess as sp
import time
from typing import Dict, List, Optional, Tuple, Union

import requests
from termcolor import colored

import autogen

from autogen_prompts import *
from autogen_executor import LocalCommandLineCodeExecutor
from util import yaml_dump, flatten, parse_code2

logger = logging.getLogger(__name__)


def _config_check(config: Dict):
    # check config loading
    assert config.get("coding", None) is not None, 'Missing "coding" in your config.'
    assert config.get("default_llm_config", None) is not None, 'Missing "default_llm_config" in your config.'
    assert config.get("code_execution_config", None) is not None, 'Missing "code_execution_config" in your config.'

    for agent_config in config["agent_configs"]:
        assert agent_config.get("name", None) is not None, 'Missing agent "name" in your agent_configs.'
        assert (
            agent_config.get("system_message", None) is not None
        ), 'Missing agent "system_message" in your agent_configs.'
        assert agent_config.get("description", None) is not None, 'Missing agent "description" in your agent_configs.'


def _retrieve_json(text):
    match = re.findall(autogen.code_utils.CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
    if not match:
        return text
    code_blocks = []
    for _, code in match:
        code_blocks.append(code)
    return code_blocks[0]


def _cleanup_msg(message, include_key=True):
    key_line1 = "## Your role"
    key_line2 = "## Useful instructions for task-solving"
    key_line3 = "## Insight discovered"
    lines = message.splitlines(); new_lines = []; flag = False
    for line in lines:
        if line.startswith(key_line1) or line.startswith(key_line2) or \
            line.startswith(key_line3):
            flag = True
            if not include_key: continue
        if flag is True: new_lines.append(line)
    if len(new_lines) == 0:
        return message
    else:
        return "\n".join(new_lines)


class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, LocalCommandLineCodeExecutor):
                    return None
                return super().default(obj)

class AgentBuilder:
    """
    AgentBuilder can help user build an automatic task solving process powered by multi-agent system.
    Specifically, our building pipeline includes initialize and build.
    """

    online_server_name = "online"

    DEFAULT_PROXY_AUTO_REPLY = 'There is no code from the last 1 message for me to execute. Group chat manager should let other participants to continue the conversation. If the group chat manager want to end the conversation, you should let other participant reply me only with "TERMINATE"'

    # DEFAULT_PROXY_AUTO_REPLY = 'There is no code from the last 1 message for me to execute. Group chat manager should let other participants continue the conversation. Participants should only reply with "TERMINATE" when they have confirmed that the original problem has been fully resolved and a working solution has been verified.'

    GROUP_CHAT_DESCRIPTION = """# Group chat instruction
You are now working in a group chat with different expert and a group chat manager.
You should refer to the previous message from other participant members or yourself, follow their topic and reply to them.

**Your role is**: {name}
Group chat members: {members}{user_proxy_desc}

When the task is complete and the result has been carefully verified, after obtaining agreement from the other members, you can end the conversation by replying only with "TERMINATE".

# Your profile
{sys_msg}
"""

#     GROUP_CHAT_DESCRIPTION = """# Group chat instruction
# You are now working in a group chat with different experts and a group chat manager.
# You should refer to the previous messages from other participant members or yourself, follow their topic and reply to them.

# **Your role is**: {name}
# Group chat members: {members}{user_proxy_desc}

# The conversation should only be terminated under these conditions:
# 1. The task has been fully completed
# 2. The solution has been thoroughly tested and verified to be correct
# 3. All group members have explicitly confirmed the solution works as intended
# 4. There is clear consensus among participants that no further improvements are needed

# Once all these conditions are met, you can end the conversation by replying only with "TERMINATE".

# # Your profile
# {sys_msg}
# """

    DEFAULT_DESCRIPTION = """## Your role
[Complete this part with expert's name and skill description]

## Task and skill instructions
- [Complete this part with task description]
- [Complete this part with skill description]
- [(Optional) Complete this part with other information]
"""

    DEFAULT_CODING_INSTRUCTION = """## Useful instructions for task-solving
- [Complete this part with useful instructions for task solving]
## How to verify?
- [Complete this part with useful instructions for task verification]
## How to use code?
- [Complete this part with useful instructions for using the code]
- [(Optional) Complete this part with other information]
"""

# - Very important: Before writing test cases, first write the code for the task or function being tested.
# - When calling a function, ensure that the function has been properly defined first.
    CODING_AND_TASK_SKILL_INSTRUCTION = """## Useful instructions for task-solving
- Solve the task step by step if you need to.
- When you find an answer, verify the answer carefully. Include verifiable evidence with possible test case in your response if possible.
- All your reply should be based on the provided facts.

## How to verify?
**You have to keep believing that everyone else's answers are wrong until they provide clear enough evidence.**
- Verifying with step-by-step backward reasoning.
- Write test cases according to the general task.
- Ensure that calling a function does not result in a "NameError: name <function name> is not defined" exception.

## How to use code?
- Suggest python code (in a python coding block) or shell script (in a sh coding block) for the Computer_terminal to execute.
- If missing python packages, you can install the package by suggesting a `pip install` code in the ```sh ... ``` block.
- When using code, you must indicate the script type in the coding block.
- Do not the coding block which requires users to modify.
- Do not suggest a coding block if it's not intended to be executed by the Computer_terminal.
- The Computer_terminal cannot modify your code.
- **Use 'print' function for the output when relevant**.
- Check the execution result returned by the Computer_terminal.
- Do not ask Computer_terminal to copy and paste the result.
- If the result indicates there is an error, fix the error and output the code again. """

    CODING_PROMPT = """Does the following task need programming (i.e., access external API or tool by coding) to solve,
or coding may help the following task become easier?

TASK: {task}

Answer only YES or NO.
"""

    AGENT_NAME_PROMPT = """# Your task
Suggest no more than {max_agents} experts with their name according to the following user requirement.

## User requirement
{task}

# Task requirement
- Expert's name should follow the format: [skill]_Expert.
- Only reply the names of the experts, separated by ",".
- If coding skills are required, they should be limited to Python and Shell.
For example: Python_Expert, Math_Expert, ... """

    AGENT_SYS_MSG_PROMPT = """# Your goal
- According to the task and expert name, write a high-quality description for the expert by filling the given template.
- Ensure that your description are clear and unambiguous, and include all necessary information.

# Task
{task}

# Expert name
{position}

# Template
{default_sys_msg}
"""

    AGENT_DESCRIPTION_PROMPT = """# Your goal
Summarize the following expert's description in a sentence.

# Expert name
{position}

# Expert's description
{sys_msg}
"""

# New prompt added to generate custom CODING_AND_TASK_SKILL_INSTRUCTION
    AGENT_CODING_INSTRUCTION_PROMPT = """# Your goal
- According to the expert's name and description, write a high-quality list of instructions for task solving, answer verification, and using generated code.
- Use the provided template as a guide for writing the instructions.
- Ensure that your list of instructions are clear and unambiguous and include all necessary information.
- Ensure the total length of your instructions does not exceed 250 words.

# Expert name
{position}

# Expert's description
{sys_msg}

# Template
{instruct_template}
"""

    AGENT_SEARCHING_PROMPT = """# Your goal
Considering the following task, what experts should be involved to the task?

# TASK
{task}

# EXPERT LIST
{agent_list}

# Requirement
- You should consider if the experts' name and profile match the task.
- Considering the effort, you should select less then {max_agents} experts; less is better.
- Separate expert names by commas and use "_" instead of space. For example, Product_manager,Programmer
- Only return the list of expert names.
"""

    AGENT_SELECTION_PROMPT = """# Your goal
Match roles in the role set to each expert in expert set.

# Skill set
{skills}

# Expert pool (formatting with name: description)
{expert_pool}

# Answer format
```json
{{
    "skill_1 description": "expert_name: expert_description", // if there exists an expert that suitable for skill_1
    "skill_2 description": "None", // if there is no experts that suitable for skill_2
    ...
}}
```
"""

    AGENT_FUNCTION_MAP_PROMPT = """Consider the following function.
Function Name: {function_name}
Function Description: {function_description}

The agent details are given in the format: {format_agent_details}

Which one of the following agents should be able to execute this function, preferably an agent with programming background?
{agent_details}

Hint:
# Only respond with the name of the agent that is most suited to execute the function and nothing else.
"""

    UPDATED_AGENT_SYSTEM_MESSAGE = """
{agent_system_message}

You have access to execute the function: {function_name}.
With following description: {function_description}
"""

    def __init__(
        self,
        config_file_or_env: Optional[str] = "OAI_CONFIG_LIST",
        config_file_location: Optional[str] = "",
        builder_model: Optional[Union[str, list]] = [],
        agent_model: Optional[Union[str, list]] = [],
        builder_model_tags: Optional[list] = [],
        agent_model_tags: Optional[list] = [],
        max_agents: Optional[int] = 5,
        custom_coding_instruct: Optional[bool] = False,
        user_for_system_msg: Optional[bool] = False,
        code_execution_config: Optional[dict] = None,
        debug_mode: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        """
        (These APIs are experimental and may change in the future.)
        Args:
            config_file_or_env: path or environment of the OpenAI api configs.
            builder_model: specify a model as the backbone of build manager.
            agent_model: specify a model as the backbone of participant agents.
            endpoint_building_timeout: timeout for building up an endpoint server.
            max_agents: max agents for each task.
        """
        builder_model = builder_model if isinstance(builder_model, list) else [builder_model]
        builder_filter_dict = {}
        if len(builder_model) != 0:
            builder_filter_dict.update({"model": builder_model})
        if len(builder_model_tags) != 0:
            builder_filter_dict.update({"tags": builder_model_tags})
        builder_config_list = autogen.config_list_from_json(config_file_or_env, filter_dict=builder_filter_dict)
        if len(builder_config_list) == 0:
            raise RuntimeError(
                f"Fail to initialize build manager: {builder_model}{builder_model_tags} does not exist in {config_file_or_env}. "
                f'If you want to change this model, please specify the "builder_model" in the constructor.'
            )
        self.builder_model = autogen.OpenAIWrapper(
            config_list=builder_config_list,
            use_cache=use_cache)

        self.agent_model = agent_model if isinstance(agent_model, list) else [agent_model]
        self.agent_model_tags = agent_model_tags
        self.config_file_or_env = config_file_or_env
        self.config_file_location = config_file_location

        self.building_task: str = None
        self.agent_configs: List[Dict] = []
        self.open_ports: List[str] = []
        self.agent_procs: Dict[str, Tuple[sp.Popen, str]] = {}
        self.agent_procs_assign: Dict[str, Tuple[autogen.ConversableAgent, str]] = {}
        self.cached_configs: Dict = {}

        self.max_agents = max_agents
        self.custom_coding_instruct = custom_coding_instruct
        self.user_for_system_msg = user_for_system_msg
        self.debug_mode = debug_mode
        self.code_execution_config = code_execution_config

        if self.code_execution_config is None:
            self.code_execution_config = {
                "last_n_messages": 1,
                "work_dir": "groupchat",
                "use_docker": False,
                "timeout": 10,
            }

    def set_builder_model(self, model: str):
        self.builder_model = model

    def set_agent_model(self, model: str):
        self.agent_model = model

    def _get_agent_desc(
        self,
        agent_config,
        full_desc=True,
        include_insights=True,
        include_coding_instruct=True):

        name = agent_config['name']
        if not full_desc:
            return agent_config["description"]
        system_message = [agent_config["system_message"]]

        insights = ""
        if include_insights and "insights" in agent_config:
            insights = agent_config["insights"]
            insights = f"## Useful insights and experience for task-solving\n{insights}"

        if len(insights) > 0:
            system_message.append(insights)
        else:
            print(colored("Empty insights for %s" % name, "red"), flush=True)

        # if custom_coding_instruct, accept custom or default coding instructions
        agent_coding_instruct = ""
        if include_coding_instruct and self.custom_coding_instruct is True and \
            "coding_instruction" in agent_config:
            agent_coding_instruct = agent_config["coding_instruction"]
        elif include_coding_instruct: # and "coding_instruction" not in agent_config:
            agent_coding_instruct = self.CODING_AND_TASK_SKILL_INSTRUCTION
            if "coding_instruction" in agent_config:
                print("%s has 'coding_instruction' entry, but using default instead." % name, flush=True)
        # elif include_coding_instruct:
        #     raise Exception("If agents have 'coding_instruction' entry, "
        #         "set builder's custom_coding_instruct to True'")

        if len(agent_coding_instruct) > 0:
            system_message.append(agent_coding_instruct)
        else:
            print(colored("Empty coding instruction for %s" % name, "red"), flush=True)

        return "\n\n".join(system_message)

    def _create_agent(
        self,
        agent_config: Dict,
        member_name: List[str],
        llm_config: dict,
        use_oai_assistant: Optional[bool] = False,
        **kwargs
    ) -> autogen.AssistantAgent:
        """
        Create a group chat participant agent.

        If the agent rely on an open-source model, this function will automatically set up an endpoint for that agent.
        The API address of that endpoint will be "localhost:{free port}".

        Args:
            agent_config: agent's config. It should include the following information:
                1. model_name: backbone model of an agent, e.g., gpt-4-1106-preview, meta/Llama-2-70b-chat
                2. agent_name: use to identify an agent in the group chat.
                3. system_message: including persona, task solving instruction, etc.
                4. description: brief description of an agent that help group chat manager to pick the speaker.
            llm_config: specific configs for LLM (e.g., config_list, seed, temperature, ...).
            use_oai_assistant: use OpenAI assistant api instead of self-constructed agent.
            world_size: the max size of parallel tensors (in most of the cases, this is identical to the amount of GPUs).

        Returns:
            agent: a set-up agent.
        """
        model_name_or_hf_repo = agent_config.get("model", [])
        model_name_or_hf_repo = (
            model_name_or_hf_repo if isinstance(model_name_or_hf_repo, list) else [model_name_or_hf_repo]
        )
        model_tags = agent_config.get("tags", [])
        agent_name = agent_config["name"]
        description = agent_config["description"]

        # Path to the customize **ConversableAgent** class.
        model_path = agent_config.get("model_path", None)
        filter_dict = {}
        if len(model_name_or_hf_repo) > 0:
            filter_dict.update({"model": model_name_or_hf_repo})
        if len(model_tags) > 0:
            filter_dict.update({"tags": model_tags})
        config_list = autogen.config_list_from_json(
            self.config_file_or_env, file_location=self.config_file_location, filter_dict=filter_dict
        )
        if len(config_list) == 0:
            raise RuntimeError(
                f"Fail to initialize agent {agent_name}: {model_name_or_hf_repo}{model_tags} does not exist in {self.config_file_or_env}.\n"
                f'If you would like to change this model, please specify the "agent_model" in the constructor.\n'
                f"If you load configs from json, make sure the model in agent_configs is in the {self.config_file_or_env}."
            )
        server_id = self.online_server_name
        current_config = llm_config.copy()
        current_config.update({"config_list": config_list})
        sys_msg_role = "user" if self.user_for_system_msg else "system"
        if use_oai_assistant:
            from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

            agent = GPTAssistantAgent(
                name=agent_name,
                llm_config={**current_config, "assistant_id": None},
                instructions=self._get_agent_desc(agent_config),
                overwrite_instructions=False,
                role_for_system_message=sys_msg_role
            )
        else:
            user_proxy_desc = ""
            if self.cached_configs["coding"] is True:
                user_proxy_desc = (
                    "\nThe group also include a Computer_terminal to help you run the python and shell code."
                )

            model_class = autogen.AssistantAgent
            if model_path:
                module_path, model_class_name = model_path.replace("/", ".").rsplit(".", 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, model_class_name)
                if not issubclass(model_class, autogen.ConversableAgent):
                    logger.error(f"{model_class} is not a ConversableAgent. Use AssistantAgent as default")
                    model_class = autogen.AssistantAgent

            additional_config = {
                k: v
                for k, v in agent_config.items()
                if k not in ["model", "name", "system_message", "description",
                    "model_path", "tags", "coding_instruction", "insights"]
            }
            agent = model_class(
                name=agent_name, llm_config=current_config.copy(),
                description=description, **additional_config
            )

            system_message = self._get_agent_desc(agent_config)
            if len(system_message) == 0: system_message = agent.system_message
            enhanced_sys_msg = self.GROUP_CHAT_DESCRIPTION.format(
                name=agent_name, members=member_name,
                user_proxy_desc=user_proxy_desc, sys_msg=system_message
            )
            agent.update_system_message(enhanced_sys_msg)
        self.agent_procs_assign[agent_name] = (agent, server_id)
        return agent

    def clear_agent(self, agent_name: str, recycle_endpoint: Optional[bool] = True):
        """
        Clear a specific agent by name.

        Args:
            agent_name: the name of agent.
            recycle_endpoint: trigger for recycle the endpoint server. If true, the endpoint will be recycled
                when there is no agent depending on.
        """
        _, server_id = self.agent_procs_assign[agent_name]
        del self.agent_procs_assign[agent_name]
        if recycle_endpoint:
            if server_id == self.online_server_name:
                return
            else:
                for _, iter_sid in self.agent_procs_assign.values():
                    if server_id == iter_sid:
                        return
                self.agent_procs[server_id][0].terminate()
                self.open_ports.append(server_id.split("_")[-1])
        print(colored(f"Agent {agent_name} has been cleared.", "yellow"), flush=True)

    def clear_all_agents(self, recycle_endpoint: Optional[bool] = True):
        """
        Clear all cached agents.
        """
        for agent_name in [agent_name for agent_name in self.agent_procs_assign.keys()]:
            self.clear_agent(agent_name, recycle_endpoint)
        print(colored("All agents have been cleared.", "yellow"), flush=True)

    def _builder_model_create(self, messages):
        if self.debug_mode:
            os.makedirs(".debug", exist_ok=True)
            for i, message in enumerate(messages):
                msg_file = os.path.join(".debug", "msg_%s_%s" % (i, time.time()))
                # with open(msg_file, 'w') as f: json.dump(message, f, indent=4)
                yaml_dump(message, msg_file)

        return self.builder_model.create(messages=messages)

    def cleanup_code(self, code_file):
        if code_file is None: return
        assert os.path.exists(code_file)
        with open(code_file, 'r') as f: python_code = f.read()
        if len(python_code) == 0: return

        resp_code = (
            self._builder_model_create(
                messages=[
                    {
                        "role": "user",
                        "content": CLEANUP_CODE_PROMPT.format(
                            python_code=python_code
                        ),
                    }
                ]
            )
            .choices[0]
            .message.content
        )
        fixed_python_code = parse_code2(resp_code)
        if fixed_python_code is not None:
            with open(code_file, 'w') as f: f.write(fixed_python_code)

    def update_agents(
        self,
        code_generated, # Code generated by agents
        test_cases, # Test cases ran on the code
        code_performance, # Performance metrics for code
        discover_insight, # Whether to add insight to agent config
        n_agents: Optional[int] = None, # Num agents to update
        update_teamwork: Optional[bool] = False, # Improve agent synergy
        **kwargs,
    ) -> None:

        agent_configs = self.cached_configs['agent_configs']
        total_agents = len(agent_configs)
        agent_configs = random.sample(agent_configs, total_agents)
        if isinstance(test_cases, list): test_cases = "\n".join(test_cases)
        assert isinstance(test_cases, str)

        if n_agents is None:
            n_agents = total_agents
        elif n_agents == 0:
            _config_check(self.cached_configs); return
        else:
            assert 0 < n_agents <= total_agents

        if discover_insight:
            print(colored("==> Discovering agent insights...", "green"), flush=True)
            for i, agent_config in enumerate(agent_configs):
                if i >= n_agents: break

                if 'insights' not in agent_config: agent_config['insights'] = ''
                agent_name = agent_config['name']
                agent_sys_msg = agent_config['system_message']
                print(f"Preparing new insight for {agent_name}", flush=True)
                resp_agent_sys_msg = (
                    self._builder_model_create(
                        messages=[
                            {
                                "role": "user",
                                "content": AGENT_INSIGHT_PROMPT_V2.format(
                                    agent_name=agent_name,
                                    agent_sys_msg=agent_sys_msg,
                                    code_generated=code_generated,
                                    test_cases=test_cases,
                                    agent_insights=agent_config['insights']
                                ),
                            }
                        ]
                    )
                    .choices[0]
                    .message.content
                )
                agent_config['insights'] += \
                    _cleanup_msg(resp_agent_sys_msg, include_key=False) + "\n"

            _config_check(self.cached_configs); return

        print(colored("==> Updating agents...", "green"), flush=True)
        for i, agent_config in enumerate(agent_configs):
            if i >= n_agents: break

            agent_name = agent_config['name']
            agent_sys_msg = agent_config['system_message']
            print(f"Preparing updated description for {agent_name}", flush=True)
            resp_agent_sys_msg = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": UPDATE_AGENT_PROMPT.format(
                                agent_name=agent_name,
                                agent_sys_msg=agent_sys_msg,
                                default_sys_msg=self.DEFAULT_DESCRIPTION,
                                code_generated=code_generated,
                                test_cases=test_cases,
                                code_performance=code_performance
                            ),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            agent_config['system_message'] = _cleanup_msg(resp_agent_sys_msg)
            agent_sys_msg = agent_config['system_message']

            print(f"Preparing updated description summary for {agent_name}", flush=True)
            resp_agent_description = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": self.AGENT_DESCRIPTION_PROMPT.format(
                                position=agent_name,
                                sys_msg=agent_sys_msg),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            agent_config['description'] = resp_agent_description

        if update_teamwork and len(agent_configs) > 1:
            print(colored("==> Updating teamwork...", "green"), flush=True)
            for i, agent_config in enumerate(agent_configs):
                if i >= n_agents: break

                agent_name = agent_config['name']
                agent_sys_msg = agent_config['system_message']
                other_sys_msg = "\n".join(["%s\n%s" % (x['name'], x['system_message']) for x in agent_configs if x['name'] != agent_name])
                print(f"Preparing updated teamwork for {agent_name}", flush=True)
                resp_agent_sys_msg = (
                    self._builder_model_create(
                        messages=[
                            {
                                "role": "user",
                                "content": UPDATE_AGENT_TEAMWORK_PROMPT.format(
                                    agent_name=agent_name,
                                    agent_sys_msg=agent_sys_msg,
                                    default_sys_msg=self.DEFAULT_DESCRIPTION,
                                    other_sys_msg=other_sys_msg
                                ),
                            }
                        ]
                    )
                    .choices[0]
                    .message.content
                )
                agent_config['system_message'] = _cleanup_msg(resp_agent_sys_msg)

        if self.custom_coding_instruct:
            print(colored("==> Generating coding instructions...", "green"), flush=True)
            for i, agent_config in enumerate(agent_configs):
                if i >= n_agents: break

                agent_name = agent_config['name']
                print(f"Preparing updated coding instruction for {agent_name}", flush=True)
                if 'coding_instruction' in agent_config:
                    agent_coding_instruct = agent_config['coding_instruction']
                else:
                    agent_coding_instruct = self.CODING_AND_TASK_SKILL_INSTRUCTION
                resp_agent_code_instruct = (
                    self._builder_model_create(
                        messages=[
                            {
                                "role": "user",
                                "content": UPDATE_CODE_INSTRUCT_PROMPT.format(
                                    agent_name=agent_name,
                                    agent_coding_instruct=agent_coding_instruct,
                                    default_sys_msg=self.CODING_AND_TASK_SKILL_INSTRUCTION,
                                    # default_sys_msg=self.CODING_AND_TASK_SKILL_INSTRUCTION,
                                    code_generated=code_generated,
                                    test_cases=test_cases,
                                    code_performance=code_performance
                                ),
                            }
                        ]
                    )
                    .choices[0]
                    .message.content
                )
                agent_config['coding_instruction'] = _cleanup_msg(resp_agent_code_instruct)

        _config_check(self.cached_configs)

    def merge_agents(
        self,
        other_agent_configs,
        other_agent_insights=None,
        merge_insight_with_desc=False):

        assert len(other_agent_configs) > 0
        agent_configs = self.cached_configs['agent_configs']
        n_agents = set([len(x) for x in other_agent_configs])
        assert len(n_agents) == 1 and len(agent_configs) == list(n_agents)[0]

        print(colored("==> Merging agent descriptions...", "green"), flush=True)
        for i, agent_config in enumerate(agent_configs):
            agent_name = agent_config['name']
            other_names = set([x[i]['name'] for x in other_agent_configs])
            assert agent_name in other_names and len(other_names) == 1

            other_agent_sys_msgs = [x[i]['system_message'] for x in other_agent_configs]
            other_agent_sys_msgs = ["\nAgent %s description:\n%s" % (i+1, x) for i, x in enumerate(other_agent_sys_msgs)]
            other_agent_sys_msg = "\n".join(other_agent_sys_msgs)

            print(f"Preparing merged description for {agent_name}", flush=True)
            resp_agent_sys_msg = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": MERGE_AGENT_PROMPT.format(
                                agent_sys_msg=other_agent_sys_msg,
                                default_sys_msg=self.DEFAULT_DESCRIPTION
                            ),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            agent_config['system_message'] = _cleanup_msg(resp_agent_sys_msg)
            agent_sys_msg = agent_config['system_message']

            print(f"Preparing updated description summary for {agent_name}", flush=True)
            resp_agent_description = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": self.AGENT_DESCRIPTION_PROMPT.format(
                                position=agent_name,
                                sys_msg=agent_sys_msg),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            agent_config['description'] = resp_agent_description

        if self.custom_coding_instruct:
            print(colored("==> Generating coding instructions...", "green"), flush=True)
            for i, agent_config in enumerate(agent_configs):
                agent_name = agent_config['name']
                other_names = set([x[i]['name'] for x in other_agent_configs])
                assert agent_name in other_names and len(other_names) == 1

                other_agent_code_instructs = [x[i].get('coding_instruction',
                    self.CODING_AND_TASK_SKILL_INSTRUCTION) for x in other_agent_configs]
                other_agent_code_instructs = ["\nAgent %s coding instruction:\n%s" % (i+1, x) for i, x in enumerate(other_agent_code_instructs)]
                other_agent_code_instruct = "\n".join(other_agent_code_instructs)

                resp_agent_code_instruct = (
                    self._builder_model_create(
                        messages=[
                            {
                                "role": "user",
                                "content": MERGE_CODE_INSTRUCT_PROMPT.format(
                                    agent_sys_msg=other_agent_code_instruct,
                                    default_sys_msg=self.DEFAULT_CODING_INSTRUCTION,
                                    # default_sys_msg=self.CODING_AND_TASK_SKILL_INSTRUCTION,
                                ),
                            }
                        ]
                    )
                    .choices[0]
                    .message.content
                )
                agent_config['coding_instruction'] = _cleanup_msg(resp_agent_code_instruct)

        if other_agent_insights is not None:
            assert len(agent_configs) == len(other_agent_insights)
            for agent_config, other_agent_insight in zip(agent_configs, other_agent_insights):
                if other_agent_insight is None or len(other_agent_insight) == 0:
                    continue
                if merge_insight_with_desc:
                    agent_config['system_message'] += "\n\n## Useful insights and experience for task-solving\n" + other_agent_insight
                    if 'insights' in agent_config: del agent_config['insights']
                else:
                    agent_config['insights'] = other_agent_insight

        _config_check(self.cached_configs)

    def generate_agent_library(
        self,
        other_agent_configs,
        merge_insight_with_desc=False,
        library_max_size=None):

        print(colored("==> Creating agent library...", "green"), flush=True)
        assert isinstance(other_agent_configs, list)
        other_agent_configs = flatten(other_agent_configs)
        assert len(other_agent_configs) > 0; agent_set = set()

        for i, other_agent_config in enumerate(other_agent_configs):
            agent_tuple = [other_agent_config['name'], \
                other_agent_config['system_message'],
                other_agent_config['description'], None, None]
            if 'coding_instruction' in other_agent_config:
                agent_tuple[-2] = other_agent_config['coding_instruction']
            if 'insights' in other_agent_config:
                agent_tuple[-1] = other_agent_config['insights']
            agent_set.add(tuple(agent_tuple))

        # Find agents with longest description length
        agent_list = list(agent_set)
        if library_max_size is not None:
            assert library_max_size > 0
            agent_list = sorted(agent_list, key=lambda x: len(str(x)),
                reverse=True)[:library_max_size]

        other_agent_names = set(); agent_configs = []
        for i, (agent_name, agent_sys_msg, agent_desc, agent_coding, agent_insights) in enumerate(agent_list):
            print(f"Adding {agent_name} to library...", flush=True)

            agent_config = {'name': agent_name}
            if agent_name in other_agent_names:
                new_name = (
                    self._builder_model_create(
                        messages=[
                            {
                                "role": "user",
                                "content": CREATE_UNIQUE_NAME_PROMPT.format(
                                    other_agent_names=other_agent_names,
                                    agent_sys_msg=agent_sys_msg,
                                    code_generated=agent_coding,
                                    agent_insights=agent_insights
                                ),
                            }
                        ]
                    )
                    .choices[0]
                    .message.content
                )
                print(f"{agent_name} renamed to {new_name}", flush=True)
                agent_config['name'] = new_name
            else:
                new_name = agent_config['name']
            other_agent_names.add(new_name)

            agent_config['system_message'] = \
                agent_sys_msg.replace(agent_name, new_name)
            agent_config['description'] = \
                agent_desc.replace(agent_name, new_name)
            if agent_coding is not None:
                agent_config['coding_instruction'] = \
                    agent_coding.replace(agent_name, new_name)
            if agent_insights is not None:
                agent_config['insights'] = \
                    agent_insights.replace(agent_name, new_name)
            agent_configs.append(agent_config)

        if merge_insight_with_desc:
            for agent_config in agent_configs:
                if 'insights' not in agent_config:
                    continue
                agent_config['system_message'] += "\n\n## Useful insights and experience for task-solving\n" + agent_config['insights']
                del agent_config['insights']

        print(f"Added {len(agent_configs)} agents to library", flush=True)
        return agent_configs

    def build(
        self,
        building_task: str,
        default_llm_config: Dict,
        list_of_functions: Optional[List[Dict]] = None,
        coding: Optional[bool] = None,
        code_execution_config: Optional[Dict] = None,
        use_oai_assistant: Optional[bool] = False,
        user_proxy: Optional[autogen.ConversableAgent] = None,
        max_agents: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[autogen.ConversableAgent], Dict]:
        """
        Auto build agents based on the building task.

        Args:
            building_task: instruction that helps build manager (gpt-4) to decide what agent should be built.
            coding: use to identify if the user proxy (a code interpreter) should be added.
            code_execution_config: specific configs for user proxy (e.g., last_n_messages, work_dir, ...).
            default_llm_config: specific configs for LLM (e.g., config_list, seed, temperature, ...).
            list_of_functions: list of functions to be associated with Agents
            use_oai_assistant: use OpenAI assistant api instead of self-constructed agent.
            user_proxy: user proxy's class that can be used to replace the default user proxy.

        Returns:
            agent_list: a list of agents.
            cached_configs: cached configs.
        """
        self.building_task = building_task
        if code_execution_config is None:
            code_execution_config = self.code_execution_config
        if max_agents is None: max_agents = self.max_agents; assert max_agents > 0

        print(colored("==> Generating agents...", "green"), flush=True)
        resp_agent_name = (
            self._builder_model_create(
                messages=[
                    {
                        "role": "user",
                        "content": self.AGENT_NAME_PROMPT.format(task=building_task, max_agents=max_agents),
                    }
                ]
            )
            .choices[0]
            .message.content
        )

        agent_name_list = [agent_name.strip().replace(' ', '_').replace('.', '') \
            for agent_name in resp_agent_name.split(",")]
        if len(agent_name_list) > max_agents:
            agent_name_list = agent_name_list[:max_agents]
        print(f"{agent_name_list} are generated.", flush=True)

        print(colored("==> Generating system message...", "green"), flush=True)
        agent_sys_msg_list = []
        for name in agent_name_list:
            print(f"Preparing system message for {name}", flush=True)
            resp_agent_sys_msg = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": self.AGENT_SYS_MSG_PROMPT.format(
                                task=building_task,
                                position=name,
                                default_sys_msg=self.DEFAULT_DESCRIPTION,
                            ),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            agent_sys_msg_list.append(resp_agent_sys_msg)

        print(colored("==> Generating description...", "green"), flush=True)
        agent_description_list = []
        for name, sys_msg in list(zip(agent_name_list, agent_sys_msg_list)):
            print(f"Preparing description for {name}", flush=True)
            resp_agent_description = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": self.AGENT_DESCRIPTION_PROMPT.format(position=name, sys_msg=sys_msg),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            agent_description_list.append(resp_agent_description)

        if self.custom_coding_instruct:
            print(colored("==> Generating coding instructions...", "green"), flush=True)
            agent_coding_instruct_list = []
            for name, sys_msg in list(zip(agent_name_list, agent_sys_msg_list)):
                print(f"Preparing coding instructions for {name}", flush=True)
                resp_agent_coding_instruct = (
                    self._builder_model_create(
                        messages=[
                            {
                                "role": "user",
                                "content": self.AGENT_CODING_INSTRUCTION_PROMPT.format(
                                    position=name,
                                    sys_msg=sys_msg,
                                    instruct_template=self.CODING_AND_TASK_SKILL_INSTRUCTION),
                            }
                        ]
                    )
                    .choices[0]
                    .message.content
                )
                agent_coding_instruct_list.append(resp_agent_coding_instruct)
        else:
            agent_coding_instruct_list = [None] * len(agent_name_list)

        agent_configs = []
        for name, sys_msg, description, coding_instruction in list(zip(agent_name_list, agent_sys_msg_list, agent_description_list, agent_coding_instruct_list)):
            agent_config = {
                "name": name,
                "model": self.agent_model,
                "tags": self.agent_model_tags,
                "system_message": sys_msg,
                "description": description,
            }
            if coding_instruction is not None:
                agent_config["coding_instruction"] = coding_instruction,
            agent_configs.append(agent_config)

        if coding is None:
            resp = (
                self._builder_model_create(
                    messages=[{"role": "user", "content": self.CODING_PROMPT.format(task=building_task)}]
                )
                .choices[0]
                .message.content
            )
            coding = True if resp == "YES" else False

        self.cached_configs.update(
            {
                "building_task": building_task,
                "agent_configs": agent_configs,
                "coding": coding,
                "default_llm_config": default_llm_config,
                "code_execution_config": code_execution_config,
            }
        )

        _config_check(self.cached_configs)
        return self._build_agents(use_oai_assistant, list_of_functions,
            user_proxy=user_proxy, **kwargs)

    def _default_retrieval(
        self,
        building_task,
        agent_library,
        full_desc,
        top_k,
        embedding_model,
        max_agents,
        **kwargs):

        def _desc(agent): return self._get_agent_desc(agent, full_desc)

        skills = building_task.replace(":", " ").split("\n")
        # skills = [line.split("-", 1)[1].strip() if line.startswith("-") else line for line in lines]
        if len(skills) == 0: skills = [building_task]

        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(
            name="agent_list",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model),
        )
        collection.add(
            documents=[_desc(agent) for agent in agent_library],
            metadatas=[{"source": "agent_profile"} for _ in range(len(agent_library))],
            ids=[f"agent_{i}" for i in range(len(agent_library))],
        )
        agent_desc_list = set()
        for skill in skills:
            recall = set(collection.query(query_texts=[skill], n_results=top_k)["documents"][0])
            agent_desc_list = agent_desc_list.union(recall)

        agent_config_list = []
        for description in list(agent_desc_list):
            for agent in agent_library:
                if description == _desc(agent):
                    agent_config_list.append(agent.copy())
                    break
        chroma_client.delete_collection(collection.name)

        # double recall from the searching result
        expert_pool = [f"{agent['name']}: {_desc(agent)}" for agent in agent_config_list]
        while True:
            skill_agent_pair_json = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": self.AGENT_SELECTION_PROMPT.format(
                                skills=building_task, expert_pool=expert_pool
                            ),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            try:
                skill_agent_pair_json = _retrieve_json(skill_agent_pair_json)
                skill_agent_pair = json.loads(skill_agent_pair_json)
                break
            except Exception as e:
                print(e, flush=True)
                time.sleep(5)
                continue

        recalled_agent_config_list = []
        recalled_name_desc = []
        created_agent_config_list = []
        for skill, agent_profile in skill_agent_pair.items():
            # If no suitable agent, generate an agent
            if agent_profile == "None":
                _, agent_config_temp = self.build(
                    building_task=skill,
                    default_llm_config=default_llm_config.copy(),
                    coding=False,
                    use_oai_assistant=use_oai_assistant,
                    max_agents=1,
                )
                self.clear_agent(agent_config_temp["agent_configs"][0]["name"])
                created_agent_config_list.append(agent_config_temp["agent_configs"][0])
            else:
                if agent_profile in recalled_name_desc:
                    # prevent identical agents
                    continue
                recalled_name_desc.append(agent_profile)
                name = agent_profile.split(":")[0].strip()
                description = agent_profile.split(":")[1].strip()
                for agent in agent_config_list:
                    if name == agent["name"] and description == _desc(agent):
                        recalled_agent_config_list.append(agent.copy())

        agent_config_list = recalled_agent_config_list + created_agent_config_list
        return agent_config_list[:max_agents]

    def _llm_only_retrieval(
        self,
        building_task,
        agent_library,
        max_agents,
        min_agents,
        **kwargs):

        def _desc(agent): return self._get_agent_desc(agent,
            full_desc=True,
            include_insights=kwargs.get('include_insights', True),
            include_coding_instruct=kwargs.get('include_coding_instruct', True))
        def _format_agent_list(agent_dict):
            agent_list = sorted(agent_dict.values(), key=lambda x: x['name'])
            formatted_agent_list = []
            for agent in agent_list:
                agent_str = "Name: %s\nDescription:\n%s\n" % (agent['name'],
                    _desc(agent))
                formatted_agent_list.append(agent_str)
            return "\n\n".join(formatted_agent_list)

        agent_dict = {}
        for agent in agent_library:
            agent_dict[agent['name'].lower()] = agent

        retrieved_agents = []
        while True:
            if len(retrieved_agents) >= min_agents: break
            agent_name_resp = (
                self._builder_model_create(
                    messages=[
                        {
                            "role": "user",
                            "content": AGENT_LIBRARY_PROMPT.format(
                                task=building_task,
                                agent_list=_format_agent_list(agent_dict),
                                max_agents=max_agents,
                                min_agents=min_agents
                            ),
                        }
                    ]
                )
                .choices[0]
                .message.content
            )
            agent_names = [agent.strip() for agent in agent_name_resp.split(",")]
            for agent_name in agent_names:
                if agent_name.lower() in agent_dict:
                    retrieved_agents.append(agent_dict[agent_name.lower()])
                else:
                    print(colored("Bad agent name, cannot find: %s" % agent_name,
                        "green"), flush=True)

        return retrieved_agents[:max_agents]

    def build_from_library(
        self,
        building_task: str,
        library_list_or_json: Union[str,list],
        default_llm_config: Dict,
        default_retrieval: bool = False,
        full_desc: bool = True,
        top_k: Optional[int] = None,
        coding: Optional[bool] = True,
        code_execution_config: Optional[Dict] = None,
        use_oai_assistant: Optional[bool] = False,
        embedding_model: Optional[str] = "all-mpnet-base-v2",
        user_proxy: Optional[autogen.ConversableAgent] = None,
        max_agents: Optional[int] = None,
        min_agents: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[autogen.ConversableAgent], Dict]:
        """
        Build agents from a library.
        The library is a list of agent configs, which contains the name and system_message for each agent.
        We use a build manager to decide what agent in that library should be involved to the task.

        Args:
            building_task: instruction that helps build manager (gpt-4) to decide what agent should be built.
            library_list_or_json: path or JSON string config of agent library.
            default_retrieval: whether to use default 2-pass (embedding, llm) for agent retrieval
            default_llm_config: specific configs for LLM (e.g., config_list, seed, temperature, ...).
            full_desc: whether to use agent's sys msg, coding instruct, etc for retrieval
            top_k: retrieve the top K agents when using agent description embeddings
            coding: use to identify if the user proxy (a code interpreter) should be added.
            code_execution_config: specific configs for user proxy (e.g., last_n_messages, work_dir, ...).
            use_oai_assistant: use OpenAI assistant api instead of self-constructed agent.
            embedding_model: a Sentence-Transformers model use for embedding similarity to select agents from library.
                As reference, chromadb use "all-mpnet-base-v2" as default.
            user_proxy: user proxy's class that can be used to replace the default user proxy.

        Returns:
            agent_list: a list of agents.
            cached_configs: cached configs.
        """
        import sqlite3

        # Some system will have an unexcepted sqlite3 version.
        # Check if the user has installed pysqlite3.
        if int(sqlite3.version.split(".")[0]) < 3:
            try:
                __import__("pysqlite3")
                import sys

                sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            except Exception as e:
                raise e
        import chromadb
        from chromadb.utils import embedding_functions

        self.building_task = building_task
        if max_agents is None: max_agents = self.max_agents
        if min_agents is None: min_agents = max_agents
        assert max_agents > 0 and min_agents > 0
        if top_k is None: top_k = max_agents; assert top_k > 0
        if code_execution_config is None:
            code_execution_config = self.code_execution_config

        if isinstance(library_list_or_json, list):
            agent_library = library_list_or_json
            print("Loaded library from list", flush=True)
        else:
            try:
                agent_library = json.loads(library_list_or_json)
                print("Loaded library from json", flush=True)
            except json.decoder.JSONDecodeError:
                with open(library_list_or_json, "r") as f:
                    agent_library = json.load(f)
                print("Loaded library from file: %s" % library_list_or_json, flush=True)
            except Exception as e:
                raise e

        for agent in agent_library:
            assert isinstance(agent, dict)
            assert 'name' in agent and 'system_message' in agent

        print(colored("==> Looking for suitable agents in the library...", "green"), flush=True)
        if default_retrieval:
            agent_config_list = self._default_retrieval(
                building_task=building_task,
                agent_library=agent_library,
                full_desc=full_desc,
                top_k=top_k,
                embedding_model=embedding_model,
                max_agents=max_agents,
                **kwargs)
        else:
            agent_config_list = self._llm_only_retrieval(
                building_task=building_task,
                agent_library=agent_library,
                max_agents=max_agents,
                min_agents=min_agents,
                **kwargs)
        print(f"{', '.join([agent['name'] for agent in agent_config_list])} are selected.", flush=True)

        for agent_config in agent_config_list:
            agent_config['model'] = self.agent_model
            agent_config['tags'] = self.agent_model_tags

        if coding is None:
            resp = (
                self._builder_model_create(
                    messages=[{"role": "user", "content": self.CODING_PROMPT.format(task=building_task)}]
                )
                .choices[0]
                .message.content
            )
            coding = True if resp == "YES" else False

        self.cached_configs.update(
            {
                "building_task": building_task,
                "agent_configs": agent_config_list,
                "coding": coding,
                "default_llm_config": default_llm_config,
                "code_execution_config": code_execution_config,
            }
        )

        _config_check(self.cached_configs)
        return self._build_agents(use_oai_assistant, user_proxy=user_proxy, **kwargs)

    def _build_agents(
        self,
        use_oai_assistant: Optional[bool] = False,
        list_of_functions: Optional[List[Dict]] = None,
        user_proxy: Optional[autogen.ConversableAgent] = None,
        **kwargs,
    ) -> Tuple[List[autogen.ConversableAgent], Dict]:
        """
        Build agents with generated configs.

        Args:
            use_oai_assistant: use OpenAI assistant api instead of self-constructed agent.
            list_of_functions: list of functions to be associated to Agents
            user_proxy: user proxy's class that can be used to replace the default user proxy.

        Returns:
            agent_list: a list of agents.
            cached_configs: cached configs.
        """
        agent_configs = self.cached_configs["agent_configs"]
        default_llm_config = self.cached_configs["default_llm_config"]
        coding = self.cached_configs["coding"]
        code_execution_config = self.cached_configs["code_execution_config"]

        print(colored("==> Creating agents...", "green"), flush=True)
        for config in agent_configs:
            print(f"Creating agent {config['name']}...", flush=True)
            self._create_agent(
                agent_config=config.copy(),
                member_name=[agent["name"] for agent in agent_configs],
                llm_config=default_llm_config,
                use_oai_assistant=use_oai_assistant,
                **kwargs,
            )
        agent_list = [agent_config[0] for agent_config in self.agent_procs_assign.values()]

        if coding is True:
            print("Adding user console proxy...", flush=True)
            if user_proxy is None:
                user_proxy = autogen.UserProxyAgent(
                    name="Computer_terminal",
                    is_termination_msg=lambda x: x == "TERMINATE" or x == "TERMINATE.",
                    code_execution_config=code_execution_config,
                    human_input_mode="NEVER",
                    default_auto_reply=self.DEFAULT_PROXY_AUTO_REPLY,
                )
            agent_list = agent_list + [user_proxy]

            agent_details = []

            for agent in agent_list[:-1]:
                agent_details.append({"name": agent.name, "description": agent.description})

            if list_of_functions:
                for func in list_of_functions:
                    resp = (
                        self._builder_model_create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": self.AGENT_FUNCTION_MAP_PROMPT.format(
                                        function_name=func["name"],
                                        function_description=func["description"],
                                        format_agent_details='[{"name": "agent_name", "description": "agent description"}, ...]',
                                        agent_details=str(json.dumps(agent_details)),
                                    ),
                                }
                            ]
                        )
                        .choices[0]
                        .message.content
                    )

                    autogen.agentchat.register_function(
                        func["function"],
                        caller=self.agent_procs_assign[resp][0],
                        # Shouldn't this be last agent since that is user proxy?
                        executor=agent_list[-1], # executor=agent_list[0],
                        name=func["name"],
                        description=func["description"],
                    )

                    agents_current_system_message = [
                        agent["system_message"] for agent in agent_configs if agent["name"] == resp
                    ][0]

                    self.agent_procs_assign[resp][0].update_system_message(
                        self.UPDATED_AGENT_SYSTEM_MESSAGE.format(
                            agent_system_message=agents_current_system_message,
                            function_name=func["name"],
                            function_description=func["description"],
                        )
                    )

                    print(f"Function {func['name']} is registered to agent {resp}.", flush=True)

        return agent_list, self.cached_configs.copy()

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save building configs. If the filepath is not specific, this function will create a filename by encrypt the
        building_task string by md5 with "save_config_" prefix, and save config to the local path.

        Args:
            filepath: save path.

        Return:
            filepath: path save.
        """

        if filepath is None:
            filepath = f'./save_config_{hashlib.md5(self.building_task.encode("utf-8")).hexdigest()}.json'
        with open(filepath, "w") as save_file:
            json.dump(self.cached_configs, save_file, indent=4, cls=CustomJSONEncoder)
        print(colored(f"Building config saved to {filepath}", "green"), flush=True)

        return filepath

    def load(
        self,
        filepath: Optional[str] = None,
        config_json: Optional[str] = None,
        config_dict: Optional[dict] = None,
        use_oai_assistant: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[List[autogen.ConversableAgent], Dict]:
        """
        Load building configs and call the build function to complete building without calling online LLMs' api.

        Args:
            filepath: filepath or JSON string for the save config.
            config_json: JSON string for the save config.
            config_dict: dictionary for the save config
            use_oai_assistant: use OpenAI assistant api instead of self-constructed agent.

        Returns:
            agent_list: a list of agents.
            cached_configs: cached configs.
        """
        # load from path.
        if filepath is not None:
            print(colored(f"Loading config from {filepath}...", "green"), flush=True)
            with open(filepath) as f:
                cached_configs = json.load(f)
        # load json string.
        elif config_json is not None:
            print(colored("Loading config from JSON...", "green"), flush=True)
            cached_configs = json.loads(config_json)
        elif config_dict is not None:
            print(colored("Loading config from dictionary...", "green"), flush=True)
            cached_configs = copy.copy(config_dict)
        else:
            raise Exception("You need to specify a source to load builder from!")

        _config_check(cached_configs)

        agent_configs = cached_configs["agent_configs"]
        default_llm_config = cached_configs["default_llm_config"]
        coding = cached_configs["coding"]

        if kwargs.get("code_execution_config", None) is not None:
            # for test
            self.cached_configs.update(
                {
                    "building_task": cached_configs["building_task"],
                    "agent_configs": agent_configs,
                    "coding": coding,
                    "default_llm_config": default_llm_config,
                    "code_execution_config": kwargs["code_execution_config"],
                }
            )
            del kwargs["code_execution_config"]
            return self._build_agents(use_oai_assistant, **kwargs)
        else:
            code_execution_config = cached_configs["code_execution_config"]
            self.cached_configs.update(
                {
                    "building_task": cached_configs["building_task"],
                    "agent_configs": agent_configs,
                    "coding": coding,
                    "default_llm_config": default_llm_config,
                    "code_execution_config": code_execution_config,
                }
            )
            return self._build_agents(use_oai_assistant, **kwargs)
