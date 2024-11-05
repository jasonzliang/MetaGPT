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
import time

import autogen
from autogen import Cache
# from autogen.agentchat.contrib.agent_builder import AgentBuilder
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent
# from autogen.code_utils import extract_code

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from wrapt_timeout_decorator import *
# import timeout_decorator

from autogen_agent_builder import AgentBuilder
from alg_util import ID_LENGTH
from alg_util import randomword
from util import get_time, killtree, extract_code_from_chat, format_prompt

DEFAULT_MAIN_ROLE = \
"""Write a python function that can {instruction}.
Test the function and ensure that it performs correctly and efficiently.
Return ```python your_code_here ``` with NO other texts,
your code:
"""
CONFIG_FILE_OR_ENV = os.path.expanduser("~/.autogen/OAI_CONFIG_LIST")
CHAT_LLM_CONFIG = {"temperature": 0.1,
    "model": "gpt-4o-mini",
    "cache_seed": None}
BUILDER_LLM_CONFIG = {"temperature": 0.1,
    "builder_model": "gpt-4o",
    "agent_model": "gpt-4o-mini",
    "cache_seed": None,
    "custom_coding_instruct": False,
    "user_for_system_msg": False,
    "min_agents": 2,
    "max_agents": 4}
MIN_CHAT_HIST_LEN = 50000
MAX_CHAT_HIST_LEN = 100000
MAX_MSG_LEN = 5000
CHAT_TIMEOUT = 120
# TODO: FIX CACHING/CACHE SEED

# @timeout_decorator.timeout(CHAT_TIMEOUT, timeout_exception=TimeoutError)
@timeout(CHAT_TIMEOUT, timeout_exception=TimeoutError,
    dec_allow_eval=False, dec_hard_timeout=False, dec_mp_reset_signals=True)
def start_task(execution_task: str, agent_list: list,
    coding=True, chat_llm_config=CHAT_LLM_CONFIG, max_round=20):
    # last agent is user proxy, remove it and replace with new one
    # _agent_list = []; user_proxy = None
    # for agent in agent_list:
    #     if type(agent) != autogen.UserProxyAgent:
    #         _agent_list.append(agent)
    #     else:
    #         user_proxy = agent

    # limit out of control output
    # pprint.pprint(agent_list)
    context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageTokenLimiter(
                min_tokens=MIN_CHAT_HIST_LEN,
                max_tokens=MAX_CHAT_HIST_LEN,
                max_tokens_per_message=MAX_MSG_LEN)])
    # context_handling.add_to_agent(user_proxy)
    for agent in agent_list: context_handling.add_to_agent(agent)

    config_list = autogen.config_list_from_json(CONFIG_FILE_OR_ENV,
        filter_dict={"model": [chat_llm_config['model']]})
    group_chat = autogen.GroupChat(
        agents=agent_list,
        messages=[],
        max_round=max_round,
        # allow_repeat_speaker=agent_list,
        allow_repeat_speaker=agent_list[:-1] if coding is True else agent_list,
    )
    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": config_list, **chat_llm_config}
    )

    society_of_mind_agent = SocietyOfMindAgent(
        "society_of_mind",
        chat_manager=manager,
        llm_config={"config_list": config_list, **chat_llm_config}
    )
    code_execution_config = {
        "last_n_messages": 1,
        "timeout": 10,
        "use_docker": False,
        "work_dir": "/tmp/som_%s" % randomword(ID_LENGTH)
    }
    society_user_proxy = autogen.UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=code_execution_config,
        default_auto_reply="",
        is_termination_msg=lambda x: True,
    )
    with Cache.disk(cache_seed=None,
        cache_path_root='/tmp/cache_%s' % randomword(ID_LENGTH)) as cache:
        chat_result = society_user_proxy.initiate_chat(
            society_of_mind_agent,
            message=execution_task,
            cache=cache)

    return chat_result, manager._groupchat.messages
    # return agent_list[0].initiate_chat(manager, message=execution_task)


def init_builder(building_task=None,
    work_dir='groupchat',
    builder_cfg=None,
    builder_dict=None,
    builder_llm_config=BUILDER_LLM_CONFIG,
    clear_cache=False,
    use_builder_dict=False,
    max_agents=None):

    os.makedirs(work_dir, exist_ok=True)
    if clear_cache: os.system("rm -rf .cache")
    if builder_cfg is None:
        builder_cfg = os.path.join(work_dir, "autogen_builder_cfg.json")
    if max_agents is None:
        a = builder_llm_config['min_agents']
        b = builder_llm_config['max_agents']
        assert a > 0 and b > 0 and a <= b; max_agents = random.randint(a, b)
    else: assert max_agents > 0

    builder = AgentBuilder(
        config_file_or_env=CONFIG_FILE_OR_ENV,
        builder_model=builder_llm_config['builder_model'],
        agent_model=builder_llm_config['agent_model'],
        max_agents=max_agents,
        custom_coding_instruct=builder_llm_config['custom_coding_instruct'],
        user_for_system_msg=builder_llm_config['user_for_system_msg'])

    # hack to prevent "builder_model" error msg when running start_task
    _builder_llm_config = {'temperature': builder_llm_config['temperature'],
        'cache_seed': builder_llm_config['cache_seed']}

    if (use_builder_dict and builder_dict is None) or \
        (not use_builder_dict and not os.path.exists(builder_cfg)):

        print("init_builder: creating new builder")
        assert building_task is not None
        code_execution_config = {
            "last_n_messages": 1,
            "timeout": 10,
            "use_docker": False,
            "work_dir": work_dir
        }
        agent_list, agent_configs = builder.build(
            building_task=building_task,
            default_llm_config=_builder_llm_config,
            coding=True,
            code_execution_config=code_execution_config)
        builder_dict = copy.copy(builder.cached_configs)
    else:
        print("init_builder: using existing builder")
        if not use_builder_dict:
            assert os.path.exists(builder_cfg)
            # load previous agent configs
            with open(builder_cfg, "r") as f:
                builder_dict = json.load(f)

    # overwrite LLM model used by agents for code generation
    for agent_config in builder_dict["agent_configs"]:
        agent_config["model"] = [builder_llm_config['agent_model']]
    # overwrite LLM config used by agents for code generation
    builder_dict["default_llm_config"].update(_builder_llm_config)
    # for any agent with sys msg file, open file and update sys msg from file
    for agent_config in builder_dict["agent_configs"]:
        if 'system_message_file' in agent_config:
            assert os.path.exists(agent_config['system_message_file'])
            with open(agent_config['system_message_file'], 'r') as f:
                new_agent_sys_msg = f.read()
            agent_config['system_message'] = new_agent_sys_msg
    # overwrite working directory used by agents for code execution
    builder_dict["code_execution_config"]["work_dir"] = work_dir

    agent_list, agent_configs = builder.load(
        config_json=json.dumps(builder_dict, indent=4))

    # print("init_builder: builder dict")
    # pprint.pprint(builder_dict)

    if use_builder_dict:
        return agent_list, agent_configs, builder, builder_dict
    else:
        with open(builder_cfg, "w") as f:
            json.dump(builder_dict, f, indent=4)
        return agent_list, agent_configs, builder, builder_cfg

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
            builder_llm_config=builder_llm_config,
            work_dir=work_dir,
            use_builder_dict=True,
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
    work_dir="/tmp/eval_%s" % randomword(12),
    clear_cache=True,
    humaneval=True,
    max_agents=3,
):
    print(locals()); time.sleep(3)
    if work_dir is None: work_dir = result_dir
    building_task = "Generate a team of agents that can work together to generate code and solve programming problems. Each agent should have an interesting role and provide unique capabilities."

    agent_list, agent_configs, builder, builder_cfg = \
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
                    agent_list=agent_list,
                    coding=agent_configs["coding"])
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
