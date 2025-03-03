#!/usr/bin/env python
# coding: utf-8
import copy
import fire
import json
import os
import pprint
import random
import re
import sys
import traceback
import time
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

import autogen
from autogen import Cache, UserProxyAgent
# from autogen.agentchat.contrib.agent_builder import AgentBuilder
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
# from autogen.code_utils import extract_code

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from ruamel.yaml import YAML
from wrapt_timeout_decorator import *
# from scicode.parse.parse import extract_function_name
# import timeout_decorator

from autogen_agent_builder import AgentBuilder, CustomJSONEncoder
from autogen_captainagent import CaptainAgent
from autogen_society_of_mind import SocietyOfMindAgent
from autogen_executor import LocalCommandLineCodeExecutor
from autogen_prompts import FUNCTION_PROMPT_TEMPLATE, FUNCTION_PROMPT_TEMPLATE_V2
from alg_util import ID_LENGTH
from alg_util import randomword
from util import get_time, killtree, extract_code_from_chat, format_prompt
from util import yaml_dump, OutputRedirector
from util import load_imports_from_string, eval_function_from_string
from util import extract_name_from_function

DEFAULT_MAIN_ROLE = \
"""Write a python function that can {instruction}.
Test the function and ensure that it performs correctly and efficiently.
Return ```python your_code_here ``` with NO other texts,
your code:
"""
CONFIG_FILE_OR_ENV = os.path.expanduser("~/.autogen/OAI_CONFIG_LIST")
# CONFIG_FILE_OR_ENV = os.path.expanduser("~/.autogen/OAI_CONFIG_LIST_NIM")
if 'CONFIG_FILE_OR_ENV' in os.environ: # Overwrite config list from env
    CONFIG_FILE_OR_ENV = os.environ['CONFIG_FILE_OR_ENV']
CODE_EXECUTION_CONFIG = {
    "exec_max_output_len": 4000,
    "exec_timeout": 10,
    "exec_library_name": "code_library"}
CHAT_LLM_CFG_KEYS = ['api_key', 'base_url', 'cache', 'cache_seed', 'model', 'temperature']
CHAT_LLM_CONFIG = {"temperature": 0.1,
    "model": "gpt-4o",
    "cache_seed": None,
    # "cache": None,
    "min_hist_len": 50000,
    "max_hist_len": 100000,
    "max_msg_len": 15000,
    "use_llm_lingua": False,
    "llm_lingua_len": 20000,
    "max_round": 15,
    "max_speaker_select_retries": 9}
BUILDER_LLM_CONFIG = {"temperature": 0.9,
    "builder_model": "gpt-4o",
    "agent_model": "gpt-4o",
    "cache_seed": None,
    # "cache": None,
    "custom_coding_instruct": False,
    "user_for_system_msg": False,
    "min_agents": 3,
    "max_agents": 3,
    "use_agent_library": False}
CAPTAIN_LLM_CONFIG = {
    "nested_config": {
        "autobuild_init_config": {
            "config_file_or_env": CONFIG_FILE_OR_ENV,
            "custom_coding_instruct": True,
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
            "max_agents": 4
        },
        "autobuild_build_config": {
            # "default_llm_config": {"max_completion_tokens": 5000},
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 4000},
            "code_execution_config": "PUT_CODE_EXECUTOR_HERE",
            "coding": True,
            "library_list_or_json": None,
            "include_insights": True,
            "include_coding_instruct": True,
        },
        "group_chat_config": {"max_round": 15},
        "group_chat_llm_config": "PUT_CHAT_LLM_CONFIG_HERE",
        "max_expert_calls": 5,
        "max_turns": 5,
    }
}
CHAT_TIMEOUT = 300
# TODO: FIX CACHING/CACHE SEED


# @timeout_decorator.timeout(CHAT_TIMEOUT, timeout_exception=TimeoutError)
@timeout(CHAT_TIMEOUT, timeout_exception=TimeoutError,
    dec_allow_eval=False, dec_hard_timeout=False, dec_mp_reset_signals=True)
def start_task(execution_task: str,
    agent_list: list,
    chat_llm_config: dict = CHAT_LLM_CONFIG,
    builder: Optional[AgentBuilder] = None,
    builder_llm_config: dict = BUILDER_LLM_CONFIG,
    code_library: Optional[list] = None,
    imports: Optional[str] = None,
    log_file: Optional[str] = None,
    use_captain_agent: bool = False):

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        redirector = OutputRedirector(log_file); redirector.enable()

    if use_captain_agent:
        chat_result, chat_messages = _start_task_captain_agent(
            execution_task=execution_task,
            agent_list=agent_list,
            chat_llm_config=chat_llm_config,
            code_library=code_library,
            imports=imports,
            log_file=log_file)
    else:
        chat_result, chat_messages = _start_task_builder_agents(
            execution_task=execution_task,
            agent_list=agent_list,
            chat_llm_config=chat_llm_config,
            builder=builder,
            builder_llm_config=builder_llm_config,
            code_library=code_library,
            imports=imports,
            log_file=log_file)

    if log_file is not None:
        redirector.disable(); yaml_dump(chat_messages, log_file)
    return chat_result, chat_messages


def _start_task_captain_agent(
    execution_task: str,
    agent_list: list,
    chat_llm_config: dict = CHAT_LLM_CONFIG,
    code_library: Optional[list] = None,
    imports: Optional[str] = None,
    captain_agent_name: str = 'captain_agent',
    proxy_agent_name: str = 'captain_user_proxy',
    log_file: Optional[str] = None):

    assert len(agent_list) == 1 and agent_list[0].name == captain_agent_name
    captain_agent = agent_list[0]; captain_agent.reset()

    executor = captain_agent.executor._executor; assert executor is not None
    _register_functions_executor(executor, imports, code_library, work_dir=None)

    captain_user_proxy = UserProxyAgent(
        name=proxy_agent_name,
        human_input_mode="NEVER",
        code_execution_config=False,
        default_auto_reply="",
        is_termination_msg=lambda x: True)
    chat_result = captain_user_proxy.initiate_chat(captain_agent,
        message=execution_task)
    # chat_result = chat_result.summary

    if log_file is not None:
        sys_msg_log_file = os.path.splitext(log_file)[0] + "_sys_msg.json"
        with open(sys_msg_log_file, "w") as f:
            json.dump(captain_agent.executor.build_history, f, indent=4,
                cls=CustomJSONEncoder)
    chat_messages = captain_agent.executor.complete_chat_history
    return chat_result, chat_messages


def _start_task_builder_agents(
    execution_task: str,
    agent_list: list,
    chat_llm_config: dict = CHAT_LLM_CONFIG,
    builder: Optional[AgentBuilder] = None,
    builder_llm_config: dict = BUILDER_LLM_CONFIG,
    code_library: Optional[list] = None,
    imports: Optional[str] = None,
    log_file: Optional[str] = None):

    if builder_llm_config['use_agent_library']:
        assert builder is not None and isinstance(agent_list[0], dict)
        agent_list = _build_from_library(
            building_task=execution_task,
            agent_library=agent_list,
            builder=builder,
            builder_llm_config=builder_llm_config)

    orig_agent_sys_msgs = _register_functions(agent_list, imports, code_library)
    if log_file is not None:
        sys_msgs = [agent.system_message for agent in agent_list]
        sys_msg_log_file = os.path.splitext(log_file)[0] + "_sys_msg.txt"
        with open(sys_msg_log_file, 'w') as f:
            f.write("\n\n".join(sys_msgs))

    # Limit maximum length of chat history to not exceed context limit
    transforms = _get_chat_transforms(chat_llm_config)
    context_handling = transform_messages.TransformMessages(transforms=transforms)
    for agent in agent_list: context_handling.add_to_agent(agent)

    group_chat = autogen.GroupChat(
        agents=agent_list,
        messages=[],
        max_round=chat_llm_config['max_round'],
        max_retries_for_selecting_speaker=chat_llm_config['max_speaker_select_retries'],
        select_speaker_auto_verbose=True,
        send_introductions=True,
        allow_repeat_speaker=agent_list[:-1]) #if coding is True else agent_list)

    som_llm_config = _filter_chat_llm_config(chat_llm_config)

    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=som_llm_config)

    society_of_mind_agent = SocietyOfMindAgent(
        "society_of_mind",
        chat_manager=manager,
        llm_config=som_llm_config)

    society_user_proxy = autogen.UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        default_auto_reply="",
        is_termination_msg=lambda x: True)

    # Fail safe if SOM agent token length is over 128k context limit
    transforms = _get_som_transforms(chat_llm_config)
    context_handling = transform_messages.TransformMessages(transforms=transforms)
    context_handling.add_to_agent(society_of_mind_agent)
    context_handling.add_to_agent(society_user_proxy)

    # with Cache.disk(cache_seed=None, cache_path_root='/tmp/cache_%s_%s' % \
    #     (randomword(ID_LENGTH), time.time())) as cache:
    chat_result = society_user_proxy.initiate_chat(
        society_of_mind_agent,
        message=execution_task,
        cache=None)
    chat_messages = manager._groupchat.messages

    _restore_sys_msg(agent_list, orig_agent_sys_msgs)
    return chat_result, chat_messages


def _get_chat_transforms(chat_llm_config):
    if chat_llm_config['use_llm_lingua']:
        compression_params = {'target_token': chat_llm_config['llm_lingua_len']}
        _transforms = [transforms.TextMessageCompressor(
            text_compressor=LLMLingua(),
            min_tokens=chat_llm_config['llm_lingua_len'],
            compression_params=compression_params,
            cache=None)]
    else:
        _transforms = [transforms.MessageTokenLimiter(
            min_tokens=chat_llm_config['min_hist_len'],
            max_tokens=chat_llm_config['max_hist_len'],
            max_tokens_per_message=chat_llm_config['max_msg_len'],
            model=chat_llm_config['model'])]
    return _transforms


def _get_som_transforms(chat_llm_config):
    max_som_tokens = 120000
    _transforms = [transforms.MessageTokenLimiter(
            min_tokens=max_som_tokens,
            max_tokens=max_som_tokens,
            max_tokens_per_message=max_som_tokens,
            model=chat_llm_config['model'])]
    return _transforms


def _filter_builder_llm_config(builder_llm_config):
    _builder_llm_config = {'temperature': builder_llm_config['temperature'],
        'cache_seed': builder_llm_config['cache_seed']}

    if builder_llm_config['builder_model'].startswith("o"):
        del _builder_llm_config['temperature']
    return _builder_llm_config


def _filter_chat_llm_config(chat_llm_config):
    config_list = autogen.config_list_from_json(CONFIG_FILE_OR_ENV,
        filter_dict={"model": [chat_llm_config['model']]})
    _chat_llm_config = {}
    for key in chat_llm_config:
        if key in CHAT_LLM_CFG_KEYS:
            _chat_llm_config[key] = chat_llm_config[key]

    if _chat_llm_config['model'].startswith("o") and \
        'temperature' in _chat_llm_config:
        del _chat_llm_config['temperature']
    return {"config_list": config_list, **_chat_llm_config}


def _restore_sys_msg(agent_list, orig_agent_sys_msgs):
    if orig_agent_sys_msgs is None: return
    agent_list = [x for x in agent_list if not isinstance(x, autogen.UserProxyAgent)]
    assert len(agent_list) == len(orig_agent_sys_msgs)
    for agent, orig_sys_msg in zip(agent_list, orig_agent_sys_msgs):
        agent.update_system_message(orig_sys_msg)


def _register_functions_executor(
    executor,
    imports,
    code_library,
    work_dir=None):

    if executor is None: return None
    if work_dir is None:
        work_dir='/tmp/eval_%s_%s' % (randomword(ID_LENGTH), time.time())
    executor.reset(work_dir)
    if code_library is None or len(code_library) == 0:
        return None

    namespace = None; functions = []; loaded_code_library = []
    for i, func_dict in enumerate(code_library):
        try:
            if namespace is None:
                namespace = load_imports_from_string(imports)
            try:
                func_name = extract_name_from_function(func_dict['code'])
                if func_name != func_dict['name']:
                    print("_register_functions: extracted function name (%s) and "
                        "header (%s) do not match" % (func_name, func_dict['name']))
            except:
                func_name = func_dict['name']
            function = eval_function_from_string(namespace,
                func_dict['code'],
                func_name)

            functions.append(function); loaded_code_library.append(func_dict)
        except:
            traceback.print_exc(); print(imports); print(func_dict['code'])
            print("_register_functions: importing function %s failed!" % \
                func_dict['name'])
    if len(functions) == 0: return None

    executor._functions = functions
    func_list = [func_dict['code'] for func_dict in loaded_code_library]
    executor._setup_functions(imports=imports, func_list=func_list,
        overwrite_func_file=True)

    # function_msg = executor.format_functions_for_prompt(
    #     prompt_template=FUNCTION_PROMPT_TEMPLATE)
    # function_msg = FUNCTION_PROMPT_TEMPLATE_V2.format(
    #     module_name=executor.functions_module)

    # for agent in agent_list_noproxy:
    #     new_sys_msg = agent.system_message + "\n" + function_msg
    #     agent.update_system_message(new_sys_msg)

    # return orig_agent_sys_msgs
    return None


def _register_functions(
    agent_list,
    imports,
    code_library,
    work_dir=None):

    agent_list_noproxy = []; orig_agent_sys_msgs = []; user_proxy = None
    for agent in agent_list:
        assert isinstance(agent, autogen.ConversableAgent)
        if not isinstance(agent, autogen.UserProxyAgent):
            agent_list_noproxy.append(agent)
            orig_agent_sys_msgs.append(agent.system_message)
        else:
            assert user_proxy is None; user_proxy = agent
    assert len(agent_list_noproxy) > 0; assert user_proxy is not None

    executor = user_proxy._code_executor
    return _register_functions_executor(executor, imports, code_library, work_dir)


def _build_from_library(
    building_task,
    agent_library,
    builder,
    builder_llm_config):

    agent_list, agent_configs = builder.build_from_library(
        building_task=building_task,
        library_list_or_json=agent_library,
        default_llm_config=_filter_builder_llm_config(builder_llm_config),
        coding=True,
        include_insights=builder_llm_config.get('agent_lib_include_insights', True),
        include_coding_instruct=builder_llm_config.get('agent_lib_include_coding_instruct', True))

    return agent_list


def _code_executor(work_dir=None):
    if work_dir is None:
        work_dir='/tmp/eval_%s_%s' % (randomword(ID_LENGTH), time.time())
    executor = LocalCommandLineCodeExecutor(
        timeout=CODE_EXECUTION_CONFIG['exec_timeout'],
        max_output_len=CODE_EXECUTION_CONFIG['exec_max_output_len'],
        work_dir=work_dir,
        functions_module=CODE_EXECUTION_CONFIG['exec_library_name'])
    return executor


def init_captain_agent(
    work_dir=None,
    agent_name="captain_agent",
    chat_llm_config=CHAT_LLM_CONFIG,
    captain_llm_config=CAPTAIN_LLM_CONFIG):

    executor = _code_executor(work_dir)
    _chat_llm_config = _filter_chat_llm_config(chat_llm_config)
    captain_llm_config['nested_config']['autobuild_build_config']['code_execution_config'] = \
        {'executor': executor}
    captain_llm_config['nested_config']['group_chat_llm_config'] = _chat_llm_config

    ## build agents
    transforms = _get_chat_transforms(chat_llm_config)
    captain_agent = CaptainAgent(
        name=agent_name,
        llm_config=_chat_llm_config,
        nested_config=captain_llm_config['nested_config'],
        update_default_nested_config=False,
        code_execution_config={'executor': executor},
        agent_config_save_path=None,
        transforms=transforms)
    return captain_agent, executor


def init_builder(
    building_task=None,
    use_builder_dict=False,
    work_dir=None,
    builder_cfg=None,
    builder_dict=None,
    builder_llm_config=BUILDER_LLM_CONFIG,
    max_agents=None,
    clear_cache=False,
    debug_mode=False):

    if clear_cache and os.path.exists(".cache"):
        os.system("rm -rf .cache")
    if max_agents is None:
        a = builder_llm_config['min_agents']
        b = builder_llm_config['max_agents']
        assert a > 0 and b > 0 and a <= b; max_agents = random.randint(a, b)
    else: assert max_agents > 0

    executor = _code_executor(work_dir)
    builder = AgentBuilder(
        config_file_or_env=CONFIG_FILE_OR_ENV,
        builder_model=builder_llm_config['builder_model'],
        agent_model=builder_llm_config['agent_model'],
        max_agents=max_agents,
        custom_coding_instruct=builder_llm_config['custom_coding_instruct'],
        user_for_system_msg=builder_llm_config['user_for_system_msg'],
        code_execution_config={'executor': executor},
        use_cache=False,
        debug_mode=debug_mode)

    if builder_llm_config['use_agent_library']:
        agent_list = copy.deepcopy(builder_dict)
        assert agent_list is not None and isinstance(agent_list, list)
        assert isinstance(agent_list[0], dict)
        return agent_list, None, builder, None

    # hack to prevent "builder_model" error msg when running start_task
    _builder_llm_config = _filter_builder_llm_config(builder_llm_config)

    # if builder dict or builder cfg does not exist, build new team
    if (use_builder_dict and builder_dict is None) or \
        (not use_builder_dict and builder_cfg is None):

        print("init_builder: creating new builder")
        assert building_task is not None
        agent_list, agent_configs = builder.build(
            building_task=building_task,
            default_llm_config=_builder_llm_config,
            coding=True)
        builder_dict = builder.cached_configs
    else:
        print("init_builder: using existing builder")
        if not use_builder_dict:
            assert builder_cfg is not None and os.path.exists(builder_cfg)
            # load previous agent configs
            with open(builder_cfg, "r") as f: builder_dict = json.load(f)

    # overwrite LLM model used by agents for code generation
    for agent_config in builder_dict["agent_configs"]:
        agent_config["model"] = [builder_llm_config['agent_model']]
    # overwrite LLM config used by agents for code generation
    builder_dict["default_llm_config"].update(_builder_llm_config)
    # Make sure temperature is removed for o1 or o3 models
    if builder_llm_config['agent_model'].startswith('o') and \
        "temperature" in builder_dict["default_llm_config"]:
        del builder_dict["default_llm_config"]["temperature"]
    # for any agent with sys msg file, open file and update sys msg from file
    for agent_config in builder_dict["agent_configs"]:
        if 'system_message_file' in agent_config:
            sys_msg_file = agent_config['system_message_file']
            assert os.path.exists(sys_msg_file)
            print("init_builder: loading agent sys msg file: %s" % sys_msg_file)
            with open(sys_msg_file, 'r') as f: new_agent_sys_msg = f.read()
            agent_config['system_message'] = new_agent_sys_msg
            del agent_config['system_message_file']
    # overwrite working directory used by agents for code execution
    # builder_dict["code_execution_config"]["work_dir"] = work_dir

    # overwrite code execution config to use executor for code blocks in future chat
    agent_list, agent_configs = builder.load(config_dict=builder_dict,
        code_execution_config={'executor': executor})

    if use_builder_dict:
        return agent_list, agent_configs, builder, builder_dict, executor
    else:
        if builder_cfg is None:
            builder_cfg = os.path.join(work_dir, "autogen_builder_cfg.json")
        builder.save(builder_cfg)
        return agent_list, agent_configs, builder, builder_cfg, executor


def _parse_builder_cfgs(builder_cfgs, eval_mode=False):
    builder_strs = []
    if eval_mode:
        for builder_dict in builder_cfgs:
            assert isinstance(builder_dict, dict)
            if 'building_task' in builder_dict:
                builder_dict['building_task'] = ''
            builder_strs.append(json.dumps(builder_dict))
    else:
        for builder_cfg in builder_cfgs:
            assert isinstance(builder_cfg, str)
            if os.path.exists(builder_cfg):
                with open(builder_cfg, "r") as f:
                    builder_dict = json.load(f)
                if 'building_task' in builder_dict:
                    builder_dict['building_task'] = ''
                builder_str = json.dumps(builder_dict, indent=4)
            else:
                builder_str = builder_cfg
            builder_strs.append(builder_str)
    return builder_strs


def autogen_mutate(
    builder_cfg="autogen_builder_cfg.json",
    output_cfg="autogen_mutate.json",
    work_dir='groupchat',
    builder_llm_config=BUILDER_LLM_CONFIG,
    eval_mode=False):

    builder_str = _parse_builder_cfgs([builder_cfg], eval_mode=eval_mode)[0]
    mutate_prompt = \
"""Here is a JSON string that describes an existing team that contains experts with different roles for generating code.

%s

Build a new and improved version of the existing team that generates more efficient, accurate, and correct code. Make sure the new team contain interesting, original, and creative experts not seen in the existing team. However, also make sure the new team is not completely different and retains relevant experts in the existing team. The size of the new team can be larger or smaller than the existing team. Make sure the description of the experts are clear, relevant, and concise.
"""

    building_task = mutate_prompt % builder_str
    if eval_mode:
        return init_builder(building_task=building_task,
            builder_dict=None,
            use_builder_dict=True,
            builder_llm_config=builder_llm_config,
            work_dir=work_dir,
            clear_cache=False)
    else:
        return init_builder(building_task=building_task,
            builder_cfg=output_cfg,
            builder_llm_config=builder_llm_config,
            work_dir=work_dir)


def autogen_crossover(
    builder_cfgs=["autogen_builder_cfg.json", "autogen_mutate.json"],
    output_cfg="autogen_crossover.json",
    work_dir='groupchat',
    builder_llm_config=BUILDER_LLM_CONFIG,
    eval_mode=False):

    builder_strs = _parse_builder_cfgs(builder_cfgs, eval_mode=eval_mode)
    crossover_prompt = \
"""Here are multiple JSON strings where each JSON describes an existing team containing experts with different roles for generating code.

%s

Combine and merge these experts to create a new and improved team for generating more efficient, accurate, and correct code. Make sure the new team contain interesting, original, and creative combination of experts chosen from the existing teams. The size of the new team can be larger or smaller than the existing teams. Make sure the description of the experts are clear, relevant, and concise.
"""

    building_task = crossover_prompt % "\n\n".join(builder_strs)
    if eval_mode:
        return init_builder(building_task=building_task,
            builder_dict=None,
            builder_llm_config=builder_llm_config,
            work_dir=work_dir,
            use_builder_dict=True,
            clear_cache=False)
    else:
        return init_builder(building_task=building_task,
            builder_cfg=output_cfg,
            builder_llm_config=builder_llm_config,
            work_dir=work_dir)


def run_evalplus(
    result_dir="results/evalplus_results_%s" % get_time(space=False),
    builder_cfg="config/autogen_builder_cfg2.json",
    work_dir='/tmp/eval_%s_%s' % (randomword(ID_LENGTH), time.time()),
    clear_cache=True,
    humaneval=True,
    max_agents=3,
):
    print(locals()); time.sleep(3)
    if work_dir is None: work_dir = result_dir
    building_task = "Generate a team of agents that can work together to generate code and solve programming problems. Each agent should have an interesting role and provide unique capabilities."

    agent_list, agent_configs, builder, builder_cfg, _ = \
        init_builder(building_task,
            work_dir=work_dir,
            builder_cfg=builder_cfg,
            clear_cache=clear_cache,
            max_agents=max_agents)
    print("Save path: %s" % builder_cfg)
    print("Agent list: %s" % agent_list)
    print("Agent configs:")
    pprint.pprint(agent_configs)
    if humaneval:
        problems = get_human_eval_plus(); eval_name = "humaneval"
    else:
        problems = get_mbpp_plus(); eval_name = "mbpp"

    for i, (task_id, problem) in enumerate(problems.items()):
        task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
        os.makedirs(task_id_dir, exist_ok=True)
        result_file = os.path.join(task_id_dir, "0.py")
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            continue

        prompt = format_prompt(prompt=DEFAULT_MAIN_ROLE,
            instruction=problem['prompt'])
        print("\n\n#### Task ID: %s, Prompt:\n%s" % (task_id, prompt))

        code = ""; n_tries = 3
        while n_tries > 0:
            try:
                chat_result, chat_messages = start_task(
                    execution_task=prompt,
                    agent_list=agent_list)
                code = extract_code_from_chat(chat_result)
                builder.clear_all_agents(recycle_endpoint=False)
                break
            except:
                builder.clear_all_agents(recycle_endpoint=False)
                n_tries -= 1

        with open(result_file, "w") as f: f.write(code)

    builder.save(os.path.join(result_dir, "autogen_builder_cfg.json"))
    os.system("evalplus.sanitize --samples %s >/dev/null" % result_dir)
    os.system("rsync -avz %s-sanitized/ %s >/dev/null" % \
        (result_dir, result_dir))
    os.system("rm -rf %s-sanitized" % result_dir)
    os.system("evalplus.evaluate --dataset %s --samples %s | tee %s"
        % (eval_name, result_dir, os.path.join(result_dir, "evalplus.txt")))
    os.system("cp %s %s" % (__file__, result_dir))

    killtree(os.getpid(), including_parent=False) # Kill all child processes


if __name__ == "__main__":
    # autogen_mutate()
    # autogen_crossover()
    # exit()

    print("Script Usage:")
    print("./autogen_builder.py")
    print("./autogen_builder.py [builder_cfg]")
    print("./autogen_builder.py [builder_cfg] [result_dir]")

    if len(sys.argv) == 1:
        run_evalplus()
    if len(sys.argv) == 2:
        run_evalplus(builder_cfg=sys.argv[1])
    elif len(sys.argv) == 3:
        run_evalplus(builder_cfg=sys.argv[1], result_dir=sys.argv[2])
