"""
Quick test script for immediate testing
Usage: python quick_test.py <image_path>
"""

import sys
import os
import cv2
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from recognition.text_recognizer import TextRecognizer
except ImportError:
    print("Error: Could not import TextRecognizer")
    print("Make sure you're running from the project root directory")
    print("Current directory:", os.getcwd())
    sys.exit(1)

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("="*80)
        print("Quick Test Script for Engineering Drawing OCR")
        print("="*80)
        print("\nUsage:")
        print("  python quick_test.py <image_path>")
        print("\nExample:")
        print("  python quick_test.py drawing.jpg")
        print("="*80)
        return
    
    image_path = sys.argv[1]
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at '{image_path}'")
        return
    
    print("="*80)
    print("ENGINEERING DRAWING OCR - QUICK TEST")
    print("="*80)
    print(f"\n📁 Image: {image_path}")
    
    # Load image
    print("\n1️⃣  Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image")
        return
    print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Initialize recognizer
    print("\n2️⃣  Initializing OCR engine...")
    try:
        recognizer = TextRecognizer(lang='en')
        print("   ✓ OCR engine ready")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return
    
    # Extract dimensions
    print("\n3️⃣  Extracting text and dimensions...")
    try:
        results = recognizer.extract_dimensions(image)
        print(f"   ✓ Detected {len(results['text_data'])} text elements")
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\n📊 Statistics:")
    print(f"   • Total text detected: {len(results['text_data'])}")
    print(f"   • Labels: {len(results['labels'])}")
    print(f"   • Values: {len(results['values'])}")
    print(f"   • Symbols: {len(results['symbols'])}")
    print(f"   • Pairs: {len(results['pairs'])}")
    
    # Show all detected text
    print(f"\n📝 All Detected Text:")
    print("-"*80)
    if results['text_data']:
        for idx, item in enumerate(results['text_data'], 1):
            print(f"   {idx:2d}. [{item['type']:6s}] {item['text']:30s} (confidence: {item['confidence']:.2f})")
    else:
        print("   (no text detected)")
    
    # Show pairs
    print(f"\n🔗 Label-Value Pairs:")
    print("-"*80)
    if results['pairs']:
        for idx, pair in enumerate(results['pairs'], 1):
            value = pair['value'] if pair['value'] else "⚠️  NO VALUE"
            print(f"   {idx:2d}. {pair['label']:30s} = {value:20s} (conf: {pair['confidence']:.2f})")
    else:
        print("   (no pairs found)")
    
    # Create output directory
    output_dir = 'outputs/quick_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    print(f"\n4️⃣  Saving results...")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save annotated image
    vis_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    vis_image = recognizer.visualize_results(image, results, vis_path)
    print(f"   ✓ Visualization: {vis_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image': image_path,
            'statistics': {
                'total_text': len(results['text_data']),
                'labels': len(results['labels']),
                'values': len(results['values']),
                'pairs': len(results['pairs'])
            },
            'pairs': [
                {
                    'label': p['label'],
                    'value': p['value'],
                    'confidence': float(p['confidence'])
                }
                for p in results['pairs']
            ],
            'all_text': [
                {
                    'id': idx,
                    'text': item['text'],
                    'type': item['type'],
                    'confidence': float(item['confidence'])
                }
                for idx, item in enumerate(results['text_data'])
            ]
        }, f, indent=2, ensure_ascii=False)
    print(f"   ✓ JSON results: {json_path}")
    
    # Save simple CSV
    csv_path = os.path.join(output_dir, f"{base_name}_pairs.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Label,Value,Confidence\n")
        for pair in results['pairs']:
            value = pair['value'] if pair['value'] else ''
            f.write(f'"{pair["label"]}","{value}",{pair["confidence"]:.3f}\n')
    print(f"   ✓ CSV pairs: {csv_path}")
    
    # Create simple dictionary
    dimensions_dict = recognizer.export_to_dict(results['pairs'])
    
    # Save dictionary as text
    dict_path = os.path.join(output_dir, f"{base_name}_dictionary.txt")
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write("EXTRACTED DIMENSIONS\n")
        f.write("="*80 + "\n\n")
        for label, value in dimensions_dict.items():
            f.write(f"{label}: {value}\n")
    print(f"   ✓ Dictionary: {dict_path}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)
    print(f"\n📂 All outputs saved to: {output_dir}/")
    print(f"\n💡 Next steps:")
    print(f"   • Check the annotated image: {base_name}_annotated.jpg")
    print(f"   • Review JSON for details: {base_name}_results.json")
    print(f"   • Open CSV in Excel: {base_name}_pairs.csv")
    print(f"\n🔧 To improve detection:")
    print(f"   • Adjust thresholds in src/recognition/text_recognizer.py")
    print(f"   • See INSTRUCTIONS.md for detailed guide")
    print("="*80)

if __name__ == "__main__":
    main()