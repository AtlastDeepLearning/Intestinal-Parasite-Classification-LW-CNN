import json
from collections import Counter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import argparse

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

    # Basic Information
    story.append(Paragraph("Basic Information", styles['Heading2']))
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
    story.append(Paragraph(f"Total number of categories: {len(data['categories'])}", styles['Normal']))
    story.append(Spacer(1, 12))
    
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

    # Annotation Information
    if 'annotations' in data:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Annotation Information", styles['Heading2']))
        story.append(Paragraph(f"Total number of annotations: {len(data['annotations'])}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Count annotations per category
        category_counts = Counter()
        for ann in data['annotations']:
            category_id = ann['category_id']
            for cat in data['categories']:
                if cat['id'] == category_id:
                    category_counts[cat['name']] += 1
                    break

        ann_data = [["Category", "Count"]]
        for category, count in category_counts.items():
            ann_data.append([category, str(count)])

        ann_table = Table(ann_data, colWidths=[4*inch, 1*inch])
        ann_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(ann_table)

    # Build the PDF
    doc.build(story)

def analyze_json_file(file_path, pdf_output=None):
    """Analyze and display the contents of a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return

    # Display basic information
    print("\n=== Basic Information ===")
    print(f"Description: {data['info']['description']}")
    print(f"Date Created: {data['info']['date_created']}")
    print(f"Version: {data['info']['version']}")

    # Display categories
    print("\n=== Categories ===")
    print(f"Total number of categories: {len(data['categories'])}")
    print("\nCategory List:")
    for category in data['categories']:
        print(f"ID: {category['id']} - {category['name']}")

    # Display image information
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

    # Display annotation information if available
    if 'annotations' in data:
        print("\n=== Annotation Information ===")
        print(f"Total number of annotations: {len(data['annotations'])}")
        
        # Count annotations per category
        category_counts = Counter()
        for ann in data['annotations']:
            category_id = ann['category_id']
            for cat in data['categories']:
                if cat['id'] == category_id:
                    category_counts[cat['name']] += 1
                    break
        
        print("\nAnnotations per category:")
        for category, count in category_counts.items():
            print(f"{category}: {count} annotations")

    # Create PDF report if output path is specified
    if pdf_output:
        try:
            create_pdf_report(data, pdf_output)
            print(f"\nPDF report has been saved to: {pdf_output}")
        except Exception as e:
            print(f"Error creating PDF report: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze JSON file and optionally create PDF report.')
    parser.add_argument('--pdf', '-p', help='Path to save the PDF report (optional)')
    args = parser.parse_args()

    file_path = "test_labels_200.json"
    analyze_json_file(file_path, args.pdf)

if __name__ == "__main__":
    main() 