import os
from markitdown import MarkItDown
from markdown_it import MarkdownIt

def convert_to_markdown(input_path, output_path):
    """
    Convert the given file to Markdown format and save it to output_path.
    Creates a folder-within-folder structure based on heading AST structure.
    """
    try:
        # Convert the file using markitdown
        markdown_content = MarkItDown(input_path)

        # Parse markdown to extract headings and content
        md = MarkdownIt()
        tokens = md.parse(markdown_content)

        # Build folder structure based on headings
        current_path = os.path.splitext(output_path)[0]
        content_buffer = []
        folder_stack = []

        for token in tokens:
            if token.type == 'heading_open':
                # When a new heading is found, flush buffer to previous folder
                if content_buffer and folder_stack:
                    folder = os.path.join(current_path, *folder_stack)
                    os.makedirs(folder, exist_ok=True)
                    with open(os.path.join(folder, 'content.md'), 'w', encoding='utf-8') as f:
                        f.write(''.join(content_buffer))
                    content_buffer = []
                # Get heading level
                level = int(token.tag[1])
                # Get heading text
                heading_text = tokens[tokens.index(token) + 1].content.strip().replace(' ', '_')
                # Adjust stack for heading level
                folder_stack = folder_stack[:level-1] + [heading_text]
            elif token.type == 'inline' or token.type == 'paragraph_open' or token.type == 'paragraph_close' or token.type.startswith('heading'):
                # Collect content
                if hasattr(token, 'content'):
                    content_buffer.append(token.content + '\n')
        # Write remaining content
        if content_buffer and folder_stack:
            folder = os.path.join(current_path, *folder_stack)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, 'content.md'), 'w', encoding='utf-8') as f:
                f.write(''.join(content_buffer))

        print(f"Converted '{input_path}' to Markdown AST folders at '{current_path}'")
    except Exception as e:
        print(f"Failed to convert '{input_path}': {e}")