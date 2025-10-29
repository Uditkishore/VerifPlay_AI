from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from rest_framework.exceptions import ValidationError
from .utils import *
from django.http import FileResponse, JsonResponse
from django.shortcuts import render
import base64
import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')
from rest_framework.exceptions import ValidationError
from bson import ObjectId
from datetime import datetime
import re




def home(request):
    """Render the home page"""
    return render(request, 'index.html')

class UvmRalGeneratorView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        # Get the uploaded Excel file from the request
        excel_file = request.FILES.get('file')

        if not excel_file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Define temporary paths for the input and output files
        excel_file_path = f"/tmp/{excel_file.name}"
        output_file = "/tmp/uvm_ral_model.sv"

        # Save the uploaded file to a temporary location
        try:
            with open(excel_file_path, 'wb') as f:
                for chunk in excel_file.chunks():
                    f.write(chunk)

            # Call the function to generate the .sv file
            try:
                excel_to_uvm_ral(excel_file_path, output_file)
            except Exception as e:
                return Response({"error": f"Error generating .sv file: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # Clean up the uploaded file
            if os.path.exists(excel_file_path):
                os.remove(excel_file_path)

        # Return the generated .sv file as a response
        if os.path.exists(output_file):
            response = FileResponse(open(output_file, 'rb'), as_attachment=True, filename="uvm_ral_model.sv")
            # Clean up the output file after returning it
            response["file-cleanup-path"] = output_file  # Adding metadata for cleanup
            return response
        else:
            return Response({"error": "Failed to generate .sv file"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UvmRalGeneratorbase64View(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        # Get the uploaded Excel file from the request
        excel_file = request.FILES.get('file')

        if not excel_file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Define temporary paths for the input and output files
        excel_file_path = f"/tmp/{excel_file.name}"
        output_file = "/tmp/uvm_ral_model.sv"

        # Save the uploaded file to a temporary location
        try:
            with open(excel_file_path, 'wb') as f:
                for chunk in excel_file.chunks():
                    f.write(chunk)

            # Call the function to generate the .sv file
            try:
                excel_to_uvm_ral(excel_file_path, output_file)
            except Exception as e:
                return Response({"error": f"Error generating .sv file: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # Clean up the uploaded file
            if os.path.exists(excel_file_path):
                os.remove(excel_file_path)

        # Return the generated .sv file as a Base64-encoded string
        if os.path.exists(output_file):
            try:
                with open(output_file, 'rb') as f:
                    file_content = f.read()
                    base64_encoded_content = base64.b64encode(file_content).decode('utf-8')
                # Clean up the output file after encoding
                os.remove(output_file)
                return Response({"file": base64_encoded_content}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": f"Error encoding file to Base64: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({"error": "Failed to generate .sv file"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DrawSystemBlockAPIView(APIView):
    def post(self, request):
        """
        Draws a system block diagram with rounded edges and improved spacing between arrows, labels, and the block.

        Args:
            input_count (int): Number of input arrows.
            output_count (int): Number of output arrows.
        """
        input_count = request.data.get('input_count')
        output_count = request.data.get('output_count')

        if not input_count:
            raise ValidationError({'message': "Input count is required"})

        if not output_count:
            raise ValidationError({'message': "Output count is required"})

        try:
            input_count = int(input_count)
            output_count = int(output_count)
        except ValueError:
            raise ValidationError({'message': "Input count and Output count must be integers"})

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
        # Save plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)


        # Return the image file as a response
        return FileResponse(buffer, as_attachment=True, filename='system_block_diagram.png')

    

class ChatAPIView(APIView):
    """
    Simplified Chat API that uses MongoDB‑stored documents
    """

    def post(self, request):
        # 1. Grab the question text
        question = request.data.get("question")

        # 2. Validate
        if not question or not isinstance(question, str):
            return Response(
                {"error": "A valid question string is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 3. Get the answer using the same retriever & llm that were
            #    initialised in utils.py
            answer = get_chatbot_response(question)

            return Response(
                {"answer": answer, "status": "success"},
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    "error": "Failed to process question",
                    "details": str(e),
                    "status": "error"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
class MuxSimulationExcelDownloadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        design_file = request.FILES.get("design_file")
        tb_file = request.FILES.get("tb_file")

        if not design_file or not tb_file:
            return Response({"error": "Both design_file and tb_file are required."}, status=400)

        result = run_mux_simulation(design_file, tb_file)

        if "error" in result:
            return Response(result, status=500)

        # Return both Excel and optional VCD path as a JSON response
        return JsonResponse({
            "message": "Simulation successful",
            "excel_file": request.build_absolute_uri("/media/mux_simulation_result.xlsx"),
            "vcd_file": request.build_absolute_uri("/media/mux_dump.vcd") if result["vcd_file"] else None,
            "stdout": result["stdout"]
        })
    
class WaveformGeneratorAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        excel_file = request.FILES.get("file")
        if not excel_file:
            return Response({"error": "Excel file is required."}, status=400)

        result = generate_waveform_from_excel(excel_file)

        if "error" in result:
            return Response(result, status=500)

        return Response(result, status=200)












    
