# -Let’s work this out in a step by step way to be sure we have the right answer.
UPDATE_AGENT_PROMPT = """# Your goal
- Write an updated high-quality description for the agent by filling the given template.
- The generated code using the agent's current description is evaluated on the test cases.
- The generated code is not correct and accuracy on the test cases can be improved.
- Use chain of thought to analyze problems with the current agent description.

# Agent name
{agent_name}

# Agent's current description
{agent_sys_msg}

# Generated code
{code_generated}

# Test Cases
{test_cases}

# Generated code accuracy
{code_performance}

# Your answer
- Let's think step by step about how to update the agent description to improve code accuracy.
- Ensure the updated agent description does not exceed 200 words and follows the template.

# Template
{default_sys_msg}
"""

MERGE_AGENT_PROMPT = """# Your goal
- Create a single description by merging multiple agent descriptions together.
- The following are the agent descriptions that generate accurate and useful code.
- Use chain of thought to analyze important/relevant elements of each agent description.

# Agent descriptions
{agent_sys_msg}

# Your answer
- Let's think step by step about merging the agent descriptions to improve code generation accuracy.
- Ensure the merged agent description does not exceed 250 words and follows the template.

# Template
{default_sys_msg}
"""

AGENT_INSIGHT_PROMPT = """# Your goal
- Write a high-quality insight for the agent by filling the given template.
- The generated code using the agent's current description is evaluated on the test cases.
- The generated code is correct and passes all the test cases.
- Use chain of thought to analyze why the current agent description is effective.

# Agent name
{agent_name}

# Agent's current description
{agent_sys_msg}

# Generated code
{code_generated}

# Test Cases
{test_cases}

# Agent's current insights
{agent_insights}

# Your answer
- Let's think step by step about an insight that explains why the current agent description is useful.
- Make sure the insight contains useful tips/knowledge that applies to similar problems.
- Make sure the insight is not a copy or restatement of any of the current insights.
- Write a single sentence summarizing the insight and be sure that it follows the template.

# Template
## Insight discovered
- [Complete this part with single sentence about any insight discovered]
"""

AGENT_INSIGHT_PROMPT_V2 = """# Your goal
- Write a high-quality insight for the agent by filling the given template.
- The generated code using the agent's current description is evaluated on the test cases.
- The generated code is correct and passes all the test cases.
- Use chain of thought to analyze why the current agent description is effective.

# Agent name
{agent_name}

# Agent's current description
{agent_sys_msg}

# Generated code
{code_generated}

# Test Cases
{test_cases}

# Agent's current insights
{agent_insights}

# Your answer
- Let's think step by step about an insight that explains how or why the agent generated correct code.
- Make sure the insight contains useful domain knowledge that applies to similar problems.
- Domain knowledge includes useful facts, principles, and theorems about science, mathematics, or programming.
- Make sure the insight is not a copy or restatement of any of the current insights.
- Write a single sentence summarizing the insight and be sure that it follows the template.

# Template
## Insight discovered
- [Complete this part with single sentence about any insight discovered]
"""

UPDATE_AGENT_TEAMWORK_PROMPT = """# Your goal
- Write an updated high-quality code description for the agent by filling the given template.
- This agent is a part of a team that works together to generate code.
- The roles and responsibilities of this agent should not overlap with that of other agents.
- Use chain of thought to analyze teamwork and synergy between current agent and rest of the team.

# Agent name
{agent_name}

# Agent's current description
{agent_sys_msg}

# Other agent names and descriptions
{other_sys_msg}

# Your answer
- Let's think step by step about how to update the agent description to improve synergy.
- Ensure the updated agent description not exceed 200 words and follows the template.

# Template
{default_sys_msg}
"""

UPDATE_CODE_INSTRUCT_PROMPT = """# Your goal
- Write an updated high-quality coding instruction for the agent by filling the given template.
- The generated code using the agent's current coding instruction is evaluated on the test cases.
- The generated code is not correct and accuracy on the test cases can be improved.
- Use chain of thought to analyze problems with the current agent coding instruction.

# Agent name
{agent_name}

# Agent's current coding instruction
{agent_coding_instruct}

# Generated code
{code_generated}

# Test Cases
{test_cases}

# Generated code accuracy
{code_performance}

# Your answer
- Let's think step by step about how to update and improve the agent's current coding instruction.
- Ensure the updated coding instruction not exceed 250 words and loosely follows the template.

# Template
{default_sys_msg}
"""

MERGE_CODE_INSTRUCT_PROMPT = """# Your goal
- Create a single coding instruction by merging multiple agent coding instructions together.
- The following are the agent coding instructions that generate accurate and useful code.
- Use chain of thought to analyze important/relevant elements of each agent coding instruction.

# Agent coding instructions
{agent_sys_msg}

# Your answer
- Let's think step by step about merging the agent coding instructions to improve code generation accuracy.
- Ensure the merged agent coding instruction does not exceed 300 words and follows the template.

# Template
{default_sys_msg}
"""

CREATE_UNIQUE_NAME_PROMPT = """# Your goal
- Create a unique name based on the agent's description, coding instruction, and insights.
- Agent's name should not be a repeat or copy of other agent's names.
- Agent's name should be relevant to the agent's description, coding instruction, and insights.
- Agent's name should be general and not be too specific or technical.

# Other agent's names
{other_agent_names}

# Agent's current description
{agent_sys_msg}

# Agent's coding instruction
{code_generated}

# Agent's insights
{agent_insights}

# Your answer
- Agent's name should follow the format: [skill]_Expert, the name must end with _Expert.
- Only answer with the name of the agent and nothing else.
For example: Python_Expert
"""

FUNCTION_PROMPT_TEMPLATE = """## How to use library functions?
- There are previously defined Python library functions in the module called '$module_name'.
- You are able to call these functions by importing them from the module '$module_name'.
- You should make use of these functions as much as possible when writing the solution code.
- For example, if there is a function named 'tensor', you can import and call it by writing:
from $module_name import tensor
def foo(args):
    result = tensor(args)
    return result

# List of functions in '$module_name'
$functions
"""

AGENT_LIBRARY_PROMPT = """# Your goal
- Considering the following task, what experts are best suited for solving the task?
- You can select from a list of experts, which contains their names and descriptions.
- Consider which experts will have the best synergy and teamwork when working together, and that their roles do not overlap.

# TASK
{task}

# START OF EXPERT LIST

{agent_list}

# END OF EXPERT LIST

# Your answer
- You should consider if the expert's name and description match the task.
- If possible, you should select {max_agents} experts and make sure each expert is unique.
- Only return a list of expert names separated by commas.
- For example: Python_Expert, Algorithm_Expert, Debugging_Expert, etc
"""
