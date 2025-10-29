import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_system_block(input_count, output_count):
    """
    Draws a system block diagram with rounded edges and improved spacing between arrows, labels, and the block.

    Args:
        input_count (int): Number of input arrows.
        output_count (int): Number of output arrows.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')

    # Block dimensions and styles
    block_width = 3
    block_height = 1.5 + max(input_count, output_count) * 0.3  # Adjust height dynamically
    block_edge_radius = 0.3
    block_facecolor = '#b3d9ff'  # Light blue
    block_edgecolor = 'black'

    # Arrow and label styles
    arrow_head_width = 0.15
    arrow_head_length = 0.3
    arrow_color = 'black'
    label_fontsize = 10
    label_padding = 0.7  # Distance between arrows and labels
    arrow_input_offset = 1.2  # Input arrow offset (increased)
    arrow_output_offset = 1.0  # Output arrow offset

    # Draw the system block (rounded rectangle)
    rect = patches.FancyBboxPatch(
        (-block_width / 2, -block_height / 2), block_width, block_height,
        boxstyle=f"round,pad=0.2,rounding_size={block_edge_radius}",
        linewidth=2, edgecolor=block_edgecolor, facecolor=block_facecolor
    )
    ax.add_patch(rect)

    # Calculate arrow spacing
    input_spacing = block_height / (input_count + 1)
    output_spacing = block_height / (output_count + 1)

    # Draw input arrows and labels
    for i in range(input_count):
        y_pos = (block_height / 2) - (i + 1) * input_spacing
        ax.arrow(
            x=-block_width / 2 - arrow_input_offset, y=y_pos,
            dx=0.5, dy=0,
            head_width=arrow_head_width, head_length=arrow_head_length,
            fc=arrow_color, ec=arrow_color
        )
        ax.text(
            x=-block_width / 2 - label_padding - arrow_input_offset, y=y_pos,
            s=f"Input {i + 1}", ha='right', va='center', fontsize=label_fontsize
        )

    # Draw output arrows and labels
    for i in range(output_count):
        y_pos = (block_height / 2) - (i + 1) * output_spacing
        ax.arrow(
            x=block_width / 2 + arrow_output_offset - 0.5, y=y_pos,
            dx=0.5, dy=0,
            head_width=arrow_head_width, head_length=arrow_head_length,
            fc=arrow_color, ec=arrow_color
        )
        ax.text(
            x=block_width / 2 + label_padding + arrow_output_offset, y=y_pos,
            s=f"Output {i + 1}", ha='left', va='center', fontsize=label_fontsize
        )

    # Adjust plot limits for better aesthetics
    ax.set_xlim(-block_width * 1.5, block_width * 1.5)
    ax.set_ylim(-block_height, block_height)

    # Remove axes for a cleaner diagram
    ax.axis('off')

    # Add a title
    plt.title("System Block Diagram", fontsize=16)

    # Display the plot
    plt.show()

# Example usage
draw_system_block(input_count=12, output_count=6)
