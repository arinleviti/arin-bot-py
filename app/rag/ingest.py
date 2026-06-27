#imports Python's built-in tool for working with file paths. The TS equivalent of an import statement
from pathlib import Path

KNOWLEDGE_DIR = Path("data/knowledge")

chunks = []  # This will hold every chunk we create, across all files

# Iterate over all Markdown files in the knowledge directory.
# file_path.read_text(encoding="utf-8") — opens the file and returns its entire contents as one string
# glob("*.md") — finds all files in the directory that match the pattern "*.md" (i.e., all Markdown files)
for file_path in KNOWLEDGE_DIR.glob("*.md"):

    # Read the file content with UTF-8 encoding, handling potential BOM
    text = file_path.read_text(encoding="utf-8-sig")  

    # Split the text into sections based on the Markdown header "##"
    sections = text.split("##") 
    # The first section is the title, remove leading and trailing whitespace.
    # lstrip removes leading whitespace and the "#" character from the title, while strip() removes any remaining leading or trailing whitespace.
    title = sections[0].lstrip("#").strip()  
    body_sections = sections[1:]  # The rest are body sections

    # repr returns a raw representation of a string, including escape characters. This is useful for debugging and understanding the exact content of a string, especially when it contains special characters or whitespace.
    for section in body_sections:
        #strip also removes \n and \r characters from the beginning and end of the string, which is useful for cleaning up text data.
        heading, _, content = section.strip().partition("\n")  # Split the section into heading and content at the first newline
        heading = heading.strip()  # Remove leading and trailing whitespace from the heading
        content = content.strip()  # Remove leading and trailing whitespace from the content

        chunk_text = f"{title} - {heading}\n{content}"  # Combine title, heading, and content into a single string
        chunks.append({
            "text": chunk_text,
            "source": file_path.name,
            "heading": heading,
        })
print(f"Total chunks created: {len(chunks)}")  # Print the total number of chunks created
print()
for chunk in chunks:
    print(chunk)
    print()