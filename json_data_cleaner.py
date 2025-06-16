import json
from pprint import pprint
from collections import Counter
import argparse
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return None

def create_pdf_report(data, output_path):
    """Create a PDF report from the JSON data."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Dataset Analysis Report", title_style))
    story.append(Spacer(1, 12))

    # Dataset Information
    story.append(Paragraph("Dataset Information", styles['Heading2']))
    info_data = [
        ["Description", data['info']['description']],
        ["Date Created", data['info']['date_created']],
        ["Version", data['info']['version']]
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))

    # Categories
    story.append(Paragraph("Categories", styles['Heading2']))
    category_data = [["ID", "Name", "Supercategory"]]
    for category in data['categories']:
        category_data.append([
            str(category['id']),
            category['name'],
            str(category['supercategory']) if category['supercategory'] else "None"
        ])
    category_table = Table(category_data, colWidths=[1*inch, 3*inch, 2*inch])
    category_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(category_table)
    story.append(Spacer(1, 20))

    # Image Information
    story.append(Paragraph("Image Information", styles['Heading2']))
    story.append(Paragraph(f"Total number of images: {len(data['images'])}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Image dimensions distribution
    dimensions = Counter()
    for img in data['images']:
        dim = f"{img['width']}x{img['height']}"
        dimensions[dim] += 1

    dim_data = [["Dimensions", "Count"]]
    for dim, count in dimensions.items():
        dim_data.append([dim, str(count)])

    dim_table = Table(dim_data, colWidths=[3*inch, 1*inch])
    dim_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(dim_table)

    # Build the PDF
    doc.build(story)

def display_categories(data):
    """Display all categories in a readable format."""
    print("\n=== Categories ===")
    for category in data['categories']:
        print(f"ID: {category['id']}")
        print(f"Name: {category['name']}")
        print(f"Supercategory: {category['supercategory']}")
        print("-" * 30)

def display_image_info(data):
    """Display information about images in the dataset."""
    print("\n=== Image Information ===")
    print(f"Total number of images: {len(data['images'])}")
    
    # Count image dimensions
    dimensions = Counter()
    for img in data['images']:
        dim = f"{img['width']}x{img['height']}"
        dimensions[dim] += 1
    
    print("\nImage dimensions distribution:")
    for dim, count in dimensions.items():
        print(f"{dim}: {count} images")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and display information from a JSON file.')
    parser.add_argument('file_path', help='Path to the JSON file to process')
    parser.add_argument('--output', '-o', help='Path to save the processed information (optional)')
    parser.add_argument('--pdf', '-p', help='Path to save the PDF report (optional)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        return
    
    # Check if file is a JSON file
    if not args.file_path.lower().endswith('.json'):
        print("Warning: The file might not be a JSON file. Make sure it contains valid JSON data.")
    
    data = load_json_file(args.file_path)
    
    if data:
        # Display dataset info
        print("\n=== Dataset Information ===")
        print(f"Description: {data['info']['description']}")
        print(f"Date Created: {data['info']['date_created']}")
        print(f"Version: {data['info']['version']}")
        
        # Display categories
        display_categories(data)
        
        # Display image information
        display_image_info(data)
        
        # If output file is specified, save the processed information
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"\nProcessed data saved to: {args.output}")
            except Exception as e:
                print(f"Error saving output file: {e}")

        # If PDF output is specified, create PDF report
        if args.pdf:
            try:
                create_pdf_report(data, args.pdf)
                print(f"\nPDF report saved to: {args.pdf}")
            except Exception as e:
                print(f"Error creating PDF report: {e}")

if __name__ == "__main__":
    main()
