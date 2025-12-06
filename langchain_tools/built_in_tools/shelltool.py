from langchain_community.tools import ShellTool

sheel_tool = ShellTool()

results = sheel_tool.invoke('whoami') # behind the scene it runs the shell command and fetch the results

print(results)

#more built in tools on langchain platform with complete description and code examples: https://python.langchain.com/en/latest/ecosystem/langchain_tools/built_in_tools/
