# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import pprint
import time
from typing import Callable, Literal, Optional, Union

from termcolor import colored

import autogen
from autogen import UserProxyAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.contrib.capabilities import transform_messages, transforms

# from autogen.agentchat.contrib.captainagent.agent_builder import AgentBuilder
from autogen.agentchat.contrib.captainagent.tool_retriever import ToolBuilder, format_ag2_tool, get_full_tool_description

from alg_util import ID_LENGTH
from alg_util import randomword
from autogen_agent_builder import AgentBuilder, CustomJSONEncoder

class CaptainAgent(ConversableAgent):
    """(In preview) Captain agent, designed to solve a task with an agent or a group of agents."""

    DEFAULT_NESTED_CONFIG = {
        "autobuild_init_config": {
            "config_file_or_env": "OAI_CONFIG_LIST",
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 2048},
            "code_execution_config": {
                "timeout": 300,
                "work_dir": "groupchat",
                "last_n_messages": 1,
                "use_docker": False,
            },
            "coding": True,
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": None,
        "max_expert_calls": 5,
        "max_turns": 6,
    }

    AUTOBUILD_TOOL = {
        "type": "function",
        "function": {
            "name": "seek_experts_help",
            "description": """Build a group of experts and let them chat with each other in a group chat.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string", "description": "[REQUIRED] Name of the group."},
                    "building_task": {
                        "type": "string",
                        "description": """Instructions that help a build manager to build a group of experts.""",
                    },
                    "execution_task": {
                        "type": "string",
                        "description": """[REQUIRED] The task that needs the experts to solve by conversation.""",
                    },
                },
            },
        },
    }

# You **must** conduct a thorough verification for the result and reason's logical compliance by leveraging the step-by-step backward reasoning with the same group of experts (using "seek_experts_help" with the same group name) when:
# - The conversation has contradictions or issues (need double-check marked as yes), or
# - The result is different from the previous results.
    AUTOBUILD_SYSTEM_MESSAGE = """# Your role
You are a perfect manager of a group of advanced experts.

# How to solve the task
When a task is assigned to you:
1. Analysis of its constraints and conditions for completion.
2. Respond with a specific plan of how to solve the task.

After that, you can solve the task in the following way:
- You are highly encouraged to seek expert help by delegating the resolution of the task to a group of relevant experts and derive conclusive insights from their conversation summarization.
- Only analyze and solve the task with your coding and language skills if you are absolutely confident that the solution can be found without the experts' help.

# How to seek experts help
The tool "seek_experts_help" can build a group of experts according to the building_task and let them chat with each other in a group chat to solve the execution_task you provided.
- This tool will summarize the essence of the experts' conversation and the derived conclusions.
- You should not modify any task information from meta_user_proxy, including code blocks, but you can provide extra information.
- Within a single response, you are limited to initiating one group of experts.

## building_task
This task helps a build manager to build a group of experts for your task.
You should suggest less then three roles (including a checker for verification) with the following format.

### Format
- [Detailed description for role 1]
- [Detailed description for role 2]
- [Detailed description for checker]

## execution_task
This is the task that needs the experts to solve by conversation.
You should Provide the following information in markdown format.

### Format
## Task description
...
## Plan for solving the task
...
## Output format
...
## Constraints and conditions for completion
...
## [Optional] results (including code blocks) and reason from last response
...

# After using "seek_experts_help"
You will receive a comprehensive conclusion from the conversation, including the task information, results, reason for the results, conversation contradiction or issues, and additional information.
You **must** conduct a thorough verification for the result and reason's logical compliance by leveraging the step-by-step backward reasoning with the same group of experts (using the tool "seek_experts_help" again with the same group name) when:
- The conversation has contradictions or issues ("Need to double-check" marked as "Yes"), or
- The result is different from the previous results.
You **must** use the tool "seek_experts_help"  again with same experts if previous result of "seek_experts_help" indicates that the answer need to be double-checked.

Note that the previous experts will forget everything after you obtain the response from them. You should provide the results (including code blocks) you collected from the previous experts' response and put it in the new execution_task.

# Some useful instructions
- You only have one tool called "seek_experts_help".
- Provide an answer yourself after "seek_experts_help".
- You should suggest python code in a python coding block (```python...```). If you need to get the value of a variable, you must use the print statement.
- When using code, you must indicate the script type in the code block.
- Do not suggest incomplete code which requires users to modify.
- Be clear about which step uses code, which step uses your language skill, and which step to build a group chat.
- If the code's result indicates there is an error, fix the error and output the whole code again.
- If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
- Include verifiable evidence in your response if possible.
- After completing and verifying all tasks, you should conclude the operation and reply "TERMINATE"
"""

    DEFAULT_DESCRIPTION = "A helpful AI assistant that can build a group of agents at a proper time to solve a task."

    # This is used to prompt the LLM to summarize the conversation history between CaptainAgent's tool execution history
    # DEFAULT_SUMMARY_PROMPT = "Read the following conversation history between an expert and a group of agent experts, summarize the conversation history. Your summarization should include the initial task, the experts' plan and the attempt, finally the results of the conversation. If the experts arrived at a conclusion, state it as it is without any modification."
    DEFAULT_SUMMARY_PROMPT = """# Your task
- A captain expert and a group of experts are working together to solve a coding problem.
- Read the following conversation history between the captain expert and group of agent experts.
- Extract the final best solution code from the discussion in the format of ```python```.
- Include only the essential implementation, removing any debugging, testing, or exploratory code.
- The solution should be complete, well-structured, and ready to use.
- Ensure the function name in the solution matches the function header name from the problem description."""

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        llm_config: Optional[Union[dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict, Literal[False]]] = False,
        nested_config: Optional[dict] = None,
        update_default_nested_config: bool = True,
        agent_lib: Optional[str] = None,
        tool_lib: Optional[str] = None,
        agent_config_save_path: Optional[str] = None,
        description: Optional[str] = DEFAULT_DESCRIPTION,
        transforms: Optional[list] = None,
        **kwargs,
    ):
        """Args:
        name (str): agent name.
        system_message (str): system message for the ChatCompletion inference.
            Please override this attribute if you want to reprogram the agent.
        llm_config (dict): llm inference configuration.
            Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create) for available options.
        is_termination_msg (function): a function that takes a message in the form of a dictionary
            and returns a boolean value indicating if this received message is a termination message.
            The dict can contain the following keys: "content", "role", "name", "function_call".
        max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
            default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
            The limit only plays a role when human_input_mode is not "ALWAYS".
        agent_lib (str): the path or a JSON file of the agent library for retrieving the nested chat instantiated by CaptainAgent.
        tool_lib (str): the path to the tool library for retrieving the tools used in the nested chat instantiated by CaptainAgent.
        nested_config (dict): the configuration for the nested chat instantiated by CaptainAgent.
            A full list of keys and their functionalities can be found in [docs](https://docs.ag2.ai/docs/topics/captainagent/configurations).
        agent_config_save_path (str): the path to save the generated or retrieved agent configuration.
        transforms (list): list of transforms to apply to the agents to limit context length.
        **kwargs (dict): Please refer to other kwargs in
            [ConversableAgent](https://github.com/ag2ai/ag2/blob/main/autogen/agentchat/conversable_agent.py#L74).
        """
        super().__init__(
            name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        if system_message is None:
            system_message = self.AUTOBUILD_SYSTEM_MESSAGE

        if update_default_nested_config:
            nested_config = self._update_config(self.DEFAULT_NESTED_CONFIG, nested_config)

        if nested_config["group_chat_llm_config"] is None:
            nested_config["group_chat_llm_config"] = llm_config.copy()
        if agent_lib:
            nested_config["autobuild_build_config"]["library_path_or_json"] = agent_lib
        if tool_lib:
            if "autobuild_tool_config" not in nested_config:
                nested_config["autobuild_tool_config"] = {}
            nested_config["autobuild_tool_config"]["tool_root"] = tool_lib

        self._system_message = system_message
        self._llm_config = llm_config
        self._nested_config = nested_config
        self._agent_config_save_path = agent_config_save_path
        self._code_execution_config = code_execution_config
        self._transforms = transforms
        self.reset()

    def reset(self):
        self.assistant = ConversableAgent(name="CaptainAgent_Expert",
            system_message=self._system_message, llm_config=self._llm_config)
        self.assistant.update_tool_signature(self.AUTOBUILD_TOOL, is_remove=False)

        self.executor = CaptainUserProxyAgent(
            name="Summoner_Expert",
            nested_config=self._nested_config,
            agent_config_save_path=self._agent_config_save_path,
            is_termination_msg=lambda x: x.get("content", "") and \
                "terminate" in x.get("content", "").lower(),
            code_execution_config=self._code_execution_config,
            human_input_mode="NEVER",
            transforms=self._transforms
        )

        self.register_nested_chats(
            [
                {
                    "sender": self.executor,
                    "recipient": self.assistant,
                    "max_turns": self._nested_config["max_turns"],
                    "summary_method": "reflection_with_llm",
                    "summary_args": {
                        "summary_prompt": self.DEFAULT_SUMMARY_PROMPT,
                    },
                }
            ],
            trigger=UserProxyAgent,
            position=0,
        )

    @staticmethod
    def _update_config(default_dict: dict, update_dict: Optional[dict]) -> dict:
        """Recursively updates the default_dict with values from update_dict."""
        if update_dict is None:
            return default_dict

        for key, value in update_dict.items():
            default_value = default_dict.get(key)
            if isinstance(default_value, dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                default_dict[key] = CaptainAgent._update_config(default_value, value)
            else:
                # Update the value or add new key
                default_dict[key] = value

        return default_dict


class CaptainUserProxyAgent(ConversableAgent):
    """(In preview) A proxy agent for the captain agent, that can execute code and provide feedback to the other agents."""

# ## Additional information (file path, code blocks, url, etc.)
# - If you found non-trivial errors or issues in the conversation, point it out with a detailed reason, and if you think it is worth further verification, mark the "Need to double-check" as "Yes".
# - If you find the conversation ends with TERMINATE and the task is solved, this is normal situation, you can mark the "Need to double-check" as "No". Only mark "No" if you are highly certain the solution is correct.
# ## Need to double-check?
# [Yes or No]
    CONVERSATION_REVIEW_PROMPT = """# Your task
- Briefly summarize the conversation history derived from an experts' group chat by following the answer format.
- If you found non-trivial errors or issues in the conversation, point it out with a detailed reason.
- Make sure "Need to double-check" is always marked as "Yes" with no extra text or explanation.
- You must output the final best solution code discovered by the experts using the ```python``` format.

# Conversation history:
{chat_history}

# Answer format
## Task
...

## Results
...

## Reason for the results
...

## Errors or issues in the conversation
...

## Need to double-check (use tool "seek_experts_help")?
{double_check}

## Final solution code
```python ...```
"""

    AUTOBUILD_TASK_DESC = """You are given: (1) a task and advises from your manager with a specific plan and (2) a general task.
Collect information from the general task, follow the suggestions from manager to solve the task.

# General Task
{general_task}

# Task and suggestions from manager
{manager_task} """

    DEFAULT_AUTO_REPLY = "I'm a proxy and I can only execute your tool or end the conversation. If you think the problem is solved, please reply me only with 'TERMINATE'."

    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
        "NEVER": "A computer terminal that can running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks), or the conversation history and result of a group of agents",
    }

    def __init__(
        self,
        name: str,
        nested_config: dict,
        agent_config_save_path: str = None,
        is_termination_msg: Optional[Callable[[dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, dict, None]] = DEFAULT_AUTO_REPLY,
        llm_config: Optional[Union[dict, Literal[False]]] = False,
        system_message: Optional[Union[str, list]] = "",
        description: Optional[str] = None,
        transforms: Optional[list] = None,
    ):
        """Args:
        name (str): name of the agent.
        nested_config (dict): the configuration for the nested chat instantiated by CaptainAgent.
        is_termination_msg (function): a function that takes a message in the form of a dictionary
            and returns a boolean value indicating if this received message is a termination message.
            The dict can contain the following keys: "content", "role", "name", "function_call".
        max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
            default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
            The limit only plays a role when human_input_mode is not "ALWAYS".
        human_input_mode (str): whether to ask for human inputs every time a message is received.
            Possible values are "ALWAYS", "TERMINATE", "NEVER".
            (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                Under this mode, the conversation stops when the human input is "exit",
                or when is_termination_msg is True and there is no human input.
            (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                the number of auto reply reaches the max_consecutive_auto_reply.
            (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
        code_execution_config (dict or False): config for the code execution.
            To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
            - work_dir (Optional, str): The working directory for the code execution.
                If None, a default working directory will be used.
                The default working directory is the "extensions" directory under
                "path_to_autogen".
            - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                Default is True, which means the code will be executed in a docker container. A default list of images will be used.
                If a list or a str of image name(s) is provided, the code will be executed in a docker container
                with the first image successfully pulled.
                If False, the code will be executed in the current environment.
                We strongly recommend using docker for code execution.
            - timeout (Optional, int): The maximum execution time in seconds.
            - last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1.
        default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.
        llm_config (dict or False): llm inference configuration.
            Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
            for available options.
            Default to false, which disables llm-based auto reply.
        system_message (str or List): system message for ChatCompletion inference.
            Only used when llm_config is not False. Use it to reprogram the agent.
        description (str): a short description of the agent. This description is used by other agents
            (e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)
        transforms (list): list of transforms to apply to the agents to limit context length.
        """
        description = (
            description if description is not None else self.DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS[human_input_mode]
        )
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=False,
            # code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
        )
        self.register_function(
            function_map={
                "seek_experts_help": lambda **args: self._run_autobuild(**args),
            }
        )
        self._agent_config_save_path = agent_config_save_path
        self._nested_config = nested_config.copy()
        self._code_execution_config = code_execution_config
        self._executor = code_execution_config.get('executor')
        self._transforms = transforms

        self.build_history = {}
        self.tool_history = {}
        self.build_times = 0
        self.complete_chat_history = []

        if self._agent_config_save_path is not None and \
            not os.path.exists(self._agent_config_save_path):
            os.makedirs(self._agent_config_save_path, exist_ok=True)

    def _run_autobuild(self, group_name: str, execution_task: str, building_task: str = "") -> str:
        """Build a group of agents by AutoBuild to solve the task.
        This function requires the nested_config to contain the autobuild_init_config, autobuild_llm_config, group_chat_llm_config.
        """
        print("==> Running AutoBuild...", flush=True)
        print("\n==> Building task: ", building_task, flush=True)
        print("\n==> Execution task: ", execution_task, flush=True)

        builder = AgentBuilder(**self._nested_config["autobuild_init_config"])
        # if the group is already built, load from history
        if group_name in self.build_history.keys():
            agent_list, agent_configs = builder.load(config_dict=self.build_history[group_name])

            # Disable tool usage
            if self._nested_config.get("autobuild_tool_config", None) \
                and agent_configs["coding"] is True:
                # tool library is enabled, reload tools and bind them to the agents
                tool_root_dir = self.tool_root_dir
                tool_builder = ToolBuilder(
                    corpus_root=tool_root_dir,
                    retriever=self._nested_config["autobuild_tool_config"].get(
                        "retriever", "all-mpnet-base-v2"),
                    type=self.tool_type,
                )
                for idx, agent in enumerate(agent_list):
                    if idx == len(self.tool_history[group_name]):
                        break
                    tool_builder.bind(agent, "\n\n".join(self.tool_history[group_name][idx]))
                agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)
        else:
            if self._nested_config["autobuild_build_config"].get("library_path_or_json", None):
                # Build from retrieval
                agent_list, agent_configs = builder.build_from_library(
                    building_task, **self._nested_config["autobuild_build_config"]
                )
                self.build_history[group_name] = agent_configs.copy()

                # Disable tool usage
                if self._nested_config.get("autobuild_tool_config", None) and agent_configs["coding"] is True:
                    skills = building_task.split("\n")
                    if len(skills) == 0:
                        skills = [building_task]

                    tool_type = "default"
                    if self._nested_config["autobuild_tool_config"].get("tool_root", "default") == "default":
                        print(colored("==> Retrieving tools...", "green"), flush=True)
                        cur_path = os.path.dirname(os.path.abspath(__file__))
                        tool_root_dir = os.path.join(cur_path, "captainagent", "tools")
                    elif isinstance(self._nested_config["autobuild_tool_config"].get("tool_root", "default"), list):
                        # We get a list, in this case, we assume it contains several tools for the agents
                        tool_root_dir = self._nested_config["autobuild_tool_config"]["tool_root"]
                        tool_type = "user_defined"
                    else:
                        tool_root_dir = self._nested_config["autobuild_tool_config"]["tool_root"]
                    self.tool_root_dir = tool_root_dir
                    self.tool_type = tool_type

                    # Retrieve and build tools based on the similarities between the skills and the tool description
                    tool_builder = ToolBuilder(
                        corpus_root=tool_root_dir,
                        retriever=self._nested_config["autobuild_tool_config"].get("retriever", "all-mpnet-base-v2"),
                        type=tool_type,
                    )
                    if tool_type == "default":
                        for idx, skill in enumerate(skills):
                            tools = tool_builder.retrieve(skill)
                            docstrings = []
                            for tool in tools:
                                category, tool_name = tool.split(" ")[0], tool.split(" ")[1]
                                tool_path = os.path.join(tool_root_dir, category, f"{tool_name}.py")
                                docstring = get_full_tool_description(tool_path)
                                docstrings.append(docstring)
                            tool_builder.bind(agent_list[idx], "\n\n".join(docstrings))
                        # the last agent is the user proxy agent, we need special treatment
                        agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)
                    else:
                        # a list containing all the tools that the agents share
                        docstrings = []
                        for tool in tool_root_dir:
                            docstrings.append(format_ag2_tool(tool))
                        for idx, agent in enumerate(agent_list):
                            if idx == len(agent_list) - 1:
                                break
                            tool_builder.bind(agent, "\n\n".join(docstrings))
                        agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)

                    # log tools
                    tool_history = self.tool_history.get(group_name, [])
                    tool_history.append(docstrings)
                    self.tool_history[group_name] = tool_history
            else:
                # Build agents from scratch
                agent_list, agent_configs = builder.build(
                    building_task, **self._nested_config["autobuild_build_config"]
                )
                self.build_history[group_name] = agent_configs.copy()

        if self._agent_config_save_path is not None:
            building_task_md5 = hashlib.md5(building_task.encode("utf-8")).hexdigest()
            with open(f"{self._agent_config_save_path}/build_history_{building_task_md5}.json",
                "w") as f:
                json.dump(self.build_history, f, indent=4, cls=CustomJSONEncoder)

        self.build_times += 1

        # Apply transforms to limit chat history context length
        context_handling = transform_messages.TransformMessages(
            transforms=self._transforms)
        for agent in agent_list: context_handling.add_to_agent(agent)

        # Start nested chat
        nested_group_chat = autogen.GroupChat(
            agents=agent_list,
            messages=[],
            allow_repeat_speaker=agent_list[:-1] if agent_configs["coding"] is True else agent_list,
            **self._nested_config["group_chat_config"],
        )
        self.manager = autogen.GroupChatManager(
            groupchat=nested_group_chat,
            llm_config=self._nested_config["group_chat_llm_config"],
        )
        key = list(self.chat_messages.keys())[0]
        general_task = self.chat_messages[key][0]["content"]

        if self._executor:
            work_dir='/tmp/eval_%s_%s' % (randomword(ID_LENGTH), time.time())
            self._executor.reset(work_dir, use_existing_func=True)

        agent_list[0].initiate_chat(
            self.manager, message=self.AUTOBUILD_TASK_DESC.format(
                general_task=general_task, manager_task=execution_task))

        if self._executor:
            work_dir='/tmp/eval_%s_%s' % (randomword(ID_LENGTH), time.time())
            self._executor.reset(work_dir, use_existing_func=True)

        chat_history = []
        key = list(agent_list[0].chat_messages.keys())[0]
        chat_messages = agent_list[0].chat_messages[key]
        for item in chat_messages:
            chat_history.append(item)
        self.complete_chat_history.extend(chat_history)

        double_check = "Yes" if self.build_times < \
            self._nested_config["max_expert_calls"] else "No"

        # Review the group chat history
        summary_model = builder.builder_model
        summarized_history = (
            summary_model.create(
                messages=[
                    {
                        "role": "user",
                        "content": self.CONVERSATION_REVIEW_PROMPT.format(
                            chat_history=chat_history,
                            double_check=double_check),
                    }
                ]
            )
            .choices[0]
            .message.content
        )

        return f"# Response from seek_experts_help: \n{summarized_history}"
