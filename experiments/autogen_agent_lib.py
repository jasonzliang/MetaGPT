#!/usr/bin/env python
# coding: utf-8

# # Automatically Build Multi-agent System from Agent Library
# 
# By: [Linxin Song](https://linxins97.github.io/), [Jieyu Zhang](https://jieyuz2.github.io/)
# 
# In this notebook, we introduce a new feature for AutoBuild, `build_from_library`, which help users build an automatic task-solving process powered by a multi-agent system from a pre-defined agent library. 
# Specifically, in `build_from_library`, we prompt an LLM to explore useful agents from a pre-defined agent library, generating configurations for those agents for a group chat to solve the user's task.

# ## Requirement
# 
# AutoBuild require `autogen-agentchat[autobuild]~=0.2`, which can be installed by the following command:

# In[1]:


# get_ipython().run_line_magic('pip', 'install autogen-agentchat[autobuild]~=0.2')


# ## Preparation and useful tools
# We need to specify a `config_path`, `default_llm_config` that include backbone LLM configurations.

# In[2]:

import copy
import fire
import json
import os
import pprint
import re
import sys
import time

import autogen

from autogen_agent_builder import AgentBuilder
# from autogen.agentchat.contrib.agent_builder import AgentBuilder


config_file_or_env = os.path.expanduser("~/.autogen/OAI_CONFIG_LIST") # modify path
llm_config = {"temperature": 0}
config_list = autogen.config_list_from_json(config_file_or_env,
    filter_dict={"model": ["gpt-4o"]})

def start_task(execution_task: str, agent_list: list):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=group_chat,
        llm_config={"config_list": config_list, **llm_config})
    agent_list[0].initiate_chat(manager, message=execution_task)


# ## Example for generating an agent library
# Here, we show an example of generating an agent library from a pre-defined list of agents' names by prompting a `gpt-4`. You can also prepare a handcrafted library yourself.
# 
# A Library contains each agent's name, description and system_message. The description is a brief introduction about agent's characteristics. As we will feed all agents' names and description to gpt-4 and let it choose the best agents for us, each agent's description should be simple but informative. 
# 
# First, we define a prompt template for description and system_message generation and a list of agents' name:

# In[3]:


AGENT_SYS_MSG_PROMPT = """Acccording to the following postion name, write a high quality instruction for the position following a given example. You should only return the instruction.

# Position Name
{position}

# Example instruction for Data Analyst

As Data Analyst, you are tasked with leveraging your extensive knowledge in data analysis to recognize and extract meaningful features from vast datasets. Your expertise in machine learning, specifically with the Random Forest Classifier, allows you to construct robust predictive models adept at handling both classification and regression tasks. You excel in model evaluation and interpretation, ensuring that the performance of your algorithms is not just assessed with precision, but also understood in the context of the data and the problem at hand. With a command over Python and proficiency in using the pandas library, you manipulate and preprocess data with ease.
"""

AGENT_DESC_PROMPT = """According to position name and the instruction, summarize the position into a high quality one sentence description.

# Position Name
{position}

# Instruction
{instruction}
"""

position_list = [
    "Environmental_Scientist",
    "Astronomer",
    "Software_Developer",
    # "Data_Analyst",
    # "Journalist",
    # "Teacher",
    # "Lawyer",
    # "Programmer",
    # "Accountant",
    # "Mathematician",
    # "Physicist",
    # "Biologist",
    # "Chemist",
    # "Statistician",
    # "IT_Specialist",
    # "Cybersecurity_Expert",
    # "Artificial_Intelligence_Engineer",
    # "Financial_Analyst",
]


# Then we can prompt a `gpt-4` model to generate each agent's system message as well as the description:

# In[4]:


build_manager = autogen.OpenAIWrapper(config_list=config_list)
sys_msg_list = []

for pos in position_list:
    resp_agent_sys_msg = (
        build_manager.create(
            messages=[
                {
                    "role": "user",
                    "content": AGENT_SYS_MSG_PROMPT.format(
                        position=pos,
                    ),
                }
            ]
        )
        .choices[0]
        .message.content
    )
    resp_desc_msg = (
        build_manager.create(
            messages=[
                {
                    "role": "user",
                    "content": AGENT_DESC_PROMPT.format(
                        position=pos,
                        instruction=resp_agent_sys_msg,
                    ),
                }
            ]
        )
        .choices[0]
        .message.content
    )
    sys_msg_list.append({"name": pos, "system_message": resp_agent_sys_msg, "description": resp_desc_msg})


# The generated profile will have the following format:

# In[5]:


print(sys_msg_list)


# We can save the generated agents' information into a json file.

# In[6]:


json.dump(sys_msg_list, open("./agent_library_example.json", "w"), indent=4)
exit()

# ## Build agents from library (by LLM)
# Here, we introduce how to build agents from the generated library. As in the previous `build`, we also need to specify a `building_task` that lets the build manager know which agents should be selected from the library according to the task. 
# 
# We also need to specify a `library_path_or_json`, which can be a path of library or a JSON string with agents' configs. Here, we use the previously saved path as the library path.

# In[7]:


library_path_or_json = "./agent_library_example.json"
building_task = "Find a paper on arxiv by programming, and analyze its application in some domain. For example, find a recent paper about gpt-4 on arxiv and find its potential applications in software."


# Then, we can call the `build_from_library` from the AgentBuilder to generate a list of agents from the library and let them complete the user's `execution_task` in a group chat.

# In[10]:


new_builder = AgentBuilder(
    config_file_or_env=config_file_or_env,
    builder_model="gpt-4o",
    agent_model="gpt-4o"
)
agent_list, _ = new_builder.build_from_library(building_task, library_path_or_json, llm_config)
start_task(
    execution_task="Find a recent paper about explainable AI on arxiv and find its potential applications in medical.",
    agent_list=agent_list,
)
new_builder.clear_all_agents()


# ## Build agents from library (by description-task similarity)
# We also support using embedding similarity to select agents. You can use a [Sentence-Transformers model](https://www.sbert.net/docs/pretrained_models.html) as an embedding extractor, and AgentBuilder will select agents with profiles that are the most similar to the building task from the library by comparing their embedding similarity. This will reduce the use of LLMs but may have less accuracy.

# In[9]:


new_builder = AgentBuilder(
    config_file_or_env=config_file_or_env,
    builder_model="gpt-4o",
    agent_model="gpt-4o"
)
agent_list, _ = new_builder.build_from_library(
    building_task, library_path_or_json, llm_config, embedding_model="all-mpnet-base-v2"
)
start_task(
    execution_task="Find a recent paper about gpt-4 on arxiv and find its potential applications in software.",
    agent_list=agent_list,
)
new_builder.clear_all_agents()

