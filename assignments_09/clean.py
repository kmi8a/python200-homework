# Save this as clean_script.py and run it in the same directory
file_path = "project_09.py"  # Replace with your actual filename

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace non-breaking space (U+00A0) with standard space (U+0020)
cleaned_content = content.replace("\xa0", " ")

with open("cleaned_script.py", "w", encoding="utf-8") as f:
    f.write(cleaned_content)

print("Non-breaking spaces successfully replaced and saved to cleaned_script.py!")