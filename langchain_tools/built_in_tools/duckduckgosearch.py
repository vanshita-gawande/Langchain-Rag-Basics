from langchain_community.tools import DuckDuckGoSearchRun

serach_tool = DuckDuckGoSearchRun()

results = serach_tool.invoke("Virat Kohli Total Odi Centuries")#behing the scene the query goes on duckduck go serach engine and fetch the results

print(results)