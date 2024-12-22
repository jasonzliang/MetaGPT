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
- For example, if there is a function named 'foo', you can import and call it by writing:
from $module_name import foo
def bar(args):
    result = foo(args)
    return result

# List of functions in '$module_name'
$functions
"""

FUNCTION_PROMPT_TEMPLATE_V2 = """## How to use library functions?
- The code from PREVIOUS PROBLEM STEPS are available as library functions.
- You are able to call these functions by importing them from the module '{module_name}'.
- You should make use of these functions to help you write the solution code.
- For example, if there is a function named 'foo', you can import and call it by writing:
from {module_name} import foo
def bar(args):
    result = foo(args)
    return result
"""

AGENT_LIBRARY_PROMPT = """# Your goal
- Considering the following task, what experts are best suited for solving the task?
- Select from a list of experts below that contains their names and detailed descriptions.
- Consider which experts will have the best synergy and teamwork when working together, and that their roles do not overlap.

# Task
{task}

# Start of expert list

{agent_list}

# End of expert list

# Your answer
- Consider if the expert's name and description match the task.
- If possible, select at least {min_agents} and at most {max_agents} different unique experts.
- Only return a list of expert names separated by commas.
- For example: Python_Expert, Algorithm_Expert, Debugging_Expert, etc
"""

CLEANUP_CODE_PROMPT = """# Your Goal
- Consider the following Python code which contains imports, functions and classes.
- Carefully check the code for any errors, mistakes, omissions, and inconsistencies, especially syntax errors.
- If there are duplicate functions or classes, choose the best implementation and remove the rest.
- Systemically find and fix all of the issues that are discovered.

# Start of Python code

{python_code}

# End of Python code

# Your answer
- Write an updated version of the Python code that fixes all issues and runs successfully without crashing.
- Ensure the updated code has all of the import, function and class names from the original code.
- Ensure the updated code includes all "# Background" comment blocks from the original code.
- Ensure your response is in the format of ```python``` with no other extraneous output.
"""

CLEANUP_CODE_PROMPT_V2 = """# Input Python code to analyze and improve:
{python_code}

# Instructions:
1. Analyze and fix any coding errors, mistakes, and inconsistencies
2. Preserve all original:
  - Import statements
  - Function/class names
3. Return output in this format only:

```python
[cleaned code here]
```

# Output requirements:
- Must execute without errors
- Maintain original functionality
- Use consistent style
"""
