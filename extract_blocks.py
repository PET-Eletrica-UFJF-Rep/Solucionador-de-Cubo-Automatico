import re
import json

file_path = r"c:\Users\eduar\Desktop\Resumo.tex"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Extract preamble (everything before \begin{document})
doc_start = content.find(r'\begin{document}')
if doc_start != -1:
    preamble = content[:doc_start + len(r'\begin{document}')]
    body = content[doc_start + len(r'\begin{document}'):]
else:
    preamble = ""
    body = content

# Placeholders
tables = {}
equations = {}

def repl_table(m):
    idx = len(tables)
    tables[idx] = m.group(0)
    return f"\n\n[TABLE_{idx}]\n\n"

def repl_eq(m):
    idx = len(equations)
    equations[idx] = m.group(0)
    return f"\n\n[EQ_{idx}]\n\n"

body = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', repl_table, body, flags=re.DOTALL)
body = re.sub(r'\\begin\{longtable\*?\}.*?\\end\{longtable\*?\}', repl_table, body, flags=re.DOTALL)
body = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', repl_eq, body, flags=re.DOTALL)

with open(r"c:\Users\eduar\Desktop\simplified.tex", "w", encoding="utf-8") as f:
    f.write(body)

with open(r"c:\Users\eduar\Desktop\blocks.json", "w", encoding="utf-8") as f:
    json.dump({"tables": tables, "equations": equations, "preamble": preamble}, f, indent=2)

print("Extraction complete. Check simplified.tex")
