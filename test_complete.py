"""
Complete testing script for engineering drawing OCR
Run this to test the entire pipeline
"""

import sys
import os
import cv2
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recognition.text_recognizer import TextRecognizer

def test_single_image(image_path, output_dir='outputs/test'):
    """
    Test OCR on a single image with detailed output
    """
    print("="*80)
    print(f"Testing image: {image_path}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize recognizer
    print("\n1. Initializing TextRecognizer...")
    recognizer = TextRecognizer(lang='en')
    
    # Load image
    print(f"2. Loading image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image")
        return None
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Extract dimensions
    print("\n3. Running OCR and text extraction...")
    results = recognizer.extract_dimensions(image)
    
    # Print statistics
    print(f"\n4. Detection Statistics:")
    print(f"   Total text detected: {len(results['text_data'])}")
    print(f"   Labels detected: {len(results['labels'])}")
    print(f"   Values detected: {len(results['values'])}")
    print(f"   Symbols detected: {len(results['symbols'])}")
    print(f"   Pairs created: {len(results['pairs'])}")
    
    # Print all detected text
    print("\n5. All Detected Text:")
    print("-"*80)
    print(f"{'Type':<10} | {'Text':<40} | {'Confidence':<10} | {'Position'}")
    print("-"*80)
    for item in results['text_data']:
        pos = f"({int(item['center'][0])}, {int(item['center'][1])})"
        print(f"{item['type']:<10} | {item['text']:<40} | {item['confidence']:<10.2f} | {pos}")
    
    # Print label-value pairs
    print("\n6. Label-Value Pairs:")
    print("-"*80)
    print(f"{'Label':<40} | {'Value':<20} | {'Confidence'}")
    print("-"*80)
    for pair in results['pairs']:
        label = pair['label']
        value = pair['value'] if pair['value'] else "❌ No value"
        confidence = pair['confidence']
        print(f"{label:<40} | {value:<20} | {confidence:.2f}")
    
    # Create visualization
    print("\n7. Creating visualization...")
    base_name = Path(image_path).stem
    
    # Save with bounding boxes
    output_vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
    vis_image = recognizer.visualize_results(image, results, output_vis_path)
    print(f"   ✓ Saved to: {output_vis_path}")
    
    # Save original with overlay
    output_overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
    overlay = create_detailed_overlay(image.copy(), results)
    cv2.imwrite(output_overlay_path, overlay)
    print(f"   ✓ Saved overlay to: {output_overlay_path}")
    
    # Save results to JSON
    print("\n8. Saving JSON results...")
    output_json_path = os.path.join(output_dir, f"{base_name}_results.json")
    save_results_json(results, image_path, output_json_path)
    print(f"   ✓ Saved to: {output_json_path}")
    
    # Save as CSV
    output_csv_path = os.path.join(output_dir, f"{base_name}_pairs.csv")
    save_pairs_csv(results['pairs'], output_csv_path)
    print(f"   ✓ Saved pairs to: {output_csv_path}")
    
    print("\n" + "="*80)
    print("✓ Testing complete!")
    print("="*80)
    
    return results

def create_detailed_overlay(image, results):
    """
    Create detailed visualization with numbered boxes
    """
    import cv2
    import numpy as np
    
    # Create semi-transparent overlay
    overlay = image.copy()
    
    for idx, item in enumerate(results['text_data']):
        bbox = np.array(item['bbox'], dtype=np.int32)
        text_type = item.get('type', 'unknown')
        
        # Color coding
        colors = {
            'label': (255, 100, 100),   # Blue
            'value': (100, 255, 100),   # Green
            'symbol': (100, 100, 255),  # Red
            'unknown': (128, 128, 128)  # Gray
        }
        color = colors.get(text_type, (128, 128, 128))
        
        # Draw filled rectangle
        cv2.fillPoly(overlay, [bbox], color)
        
        # Put number
        center = bbox.mean(axis=0).astype(int)
        cv2.putText(
            overlay,
            str(idx),
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    # Blend with original
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Draw text labels
    for idx, item in enumerate(results['text_data']):
        bbox = np.array(item['bbox'], dtype=np.int32)
        cv2.polylines(result, [bbox], True, (0, 0, 0), 2)
        
        # Add text annotation
        label_text = f"{idx}: {item['text']}"
        cv2.putText(
            result,
            label_text,
            (int(item['x_min']), int(item['y_min']) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )
    
    return result

def save_results_json(results, image_path, output_path):
    """
    Save results to JSON file
    """
    export_data = {
        'source_image': image_path,
        'statistics': {
            'total_text': len(results['text_data']),
            'labels': len(results['labels']),
            'values': len(results['values']),
            'symbols': len(results['symbols']),
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
        'all_detections': [
            {
                'id': idx,
                'text': item['text'],
                'type': item['type'],
                'confidence': float(item['confidence']),
                'bbox': [[float(x), float(y)] for x, y in item['bbox']],
                'center': [float(item['center'][0]), float(item['center'][1])]
            }
            for idx, item in enumerate(results['text_data'])
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

def save_pairs_csv(pairs, output_path):
    """
    Save label-value pairs to CSV
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Label', 'Value', 'Confidence'])
        
        for pair in pairs:
            label = pair['label']
            value = pair['value'] if pair['value'] else ''
            confidence = f"{pair['confidence']:.3f}"
            writer.writerow([label, value, confidence])

def batch_test(image_dir, output_dir='outputs/batch_test'):
    """
    Test multiple images in a directory
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    all_results = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*80}")
        print(f"Processing {idx}/{len(image_paths)}: {image_path.name}")
        print(f"{'='*80}")
        
        try:
            result = test_single_image(str(image_path), output_dir)
            all_results.append({
                'image': image_path.name,
                'success': True,
                'pairs_count': len(result['pairs']) if result else 0
            })
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {str(e)}")
            all_results.append({
                'image': image_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Save summary
    summary_path = os.path.join(output_dir, 'batch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': len(image_paths),
            'successful': sum(1 for r in all_results if r['success']),
            'failed': sum(1 for r in all_results if not r['success']),
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Batch processing complete! Summary saved to: {summary_path}")
    print(f"{'='*80}")

def compare_with_ground_truth(image_path, ground_truth_dict, output_dir='outputs/validation'):
    """
    Compare detection results with ground truth
    """
    print(f"\nValidating against ground truth...")
    
    recognizer = TextRecognizer(lang='en')
    image = cv2.imread(image_path)
    results = recognizer.extract_dimensions(image)
    
    # Convert pairs to dict
    detected = {}
    for pair in results['pairs']:
        if pair['value']:
            detected[pair['label'].lower()] = pair['value']
    
    # Compare
    correct = 0
    incorrect = 0
    missing = 0
    extra = 0
    
    print("\nComparison Results:")
    print("-"*80)
    print(f"{'Label':<30} | {'Expected':<15} | {'Detected':<15} | {'Status'}")
    print("-"*80)
    
    # Check ground truth items
    for label, expected_value in ground_truth_dict.items():
        label_lower = label.lower()
        if label_lower in detected:
            if detected[label_lower] == expected_value:
                status = "✓ Correct"
                correct += 1
            else:
                status = "✗ Incorrect"
                incorrect += 1
            detected_value = detected[label_lower]
        else:
            status = "✗ Missing"
            missing += 1
            detected_value = "N/A"
        
        print(f"{label:<30} | {expected_value:<15} | {detected_value:<15} | {status}")
    
    # Check for extra detections
    for label in detected:
        if label not in [k.lower() for k in ground_truth_dict.keys()]:
            extra += 1
            print(f"{label:<30} | {'N/A':<15} | {detected[label]:<15} | ✗ Extra")
    
    print("-"*80)
    
    # Calculate accuracy
    total = len(ground_truth_dict)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\nAccuracy Metrics:")
    print(f"  Correct: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Incorrect: {incorrect}")
    print(f"  Missing: {missing}")
    print(f"  Extra detections: {extra}")
    
    return {
        'correct': correct,
        'incorrect': incorrect,
        'missing': missing,
        'extra': extra,
        'accuracy': accuracy
    }

def main():
    """
    Main function with multiple testing modes
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Engineering Drawing OCR')
    parser.add_argument('mode', choices=['single', 'batch', 'validate'], 
                       help='Testing mode')
    parser.add_argument('--image', '-i', help='Path to single image')
    parser.add_argument('--dir', '-d', help='Directory containing images')
    parser.add_argument('--output', '-o', default='outputs', 
                       help='Output directory')
    parser.add_argument('--ground-truth', '-g', 
                       help='Path to ground truth JSON file')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.image:
            print("Error: --image required for single mode")
            print("Example: python test_complete.py single --image drawing.jpg")
            return
        
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
        
        test_single_image(args.image, args.output)
    
    elif args.mode == 'batch':
        if not args.dir:
            print("Error: --dir required for batch mode")
            print("Example: python test_complete.py batch --dir images/")
            return
        
        if not os.path.exists(args.dir):
            print(f"Error: Directory not found at {args.dir}")
            return
        
        batch_test(args.dir, args.output)
    
    elif args.mode == 'validate':
        if not args.image or not args.ground_truth:
            print("Error: --image and --ground-truth required for validate mode")
            print("Example: python test_complete.py validate --image drawing.jpg --ground-truth truth.json")
            return
        
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
        
        if not os.path.exists(args.ground_truth):
            print(f"Error: Ground truth file not found at {args.ground_truth}")
            return
        
        # Load ground truth
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
        
        compare_with_ground_truth(args.image, ground_truth, args.output)

if __name__ == "__main__":
    # If no arguments provided, show usage examples
    if len(sys.argv) == 1:
        print("="*80)
        print("Engineering Drawing OCR Test Suite")
        print("="*80)
        print("\nUsage Examples:")
        print("\n1. Test single image:")
        print("   python test_complete.py single --image path/to/drawing.jpg")
        print("\n2. Batch test multiple images:")
        print("   python test_complete.py batch --dir path/to/images/")
        print("\n3. Validate against ground truth:")
        print("   python test_complete.py validate --image drawing.jpg --ground-truth truth.json")
        print("\nOptions:")
        print("  --output, -o    Output directory (default: outputs)")
        print("\nFor more help:")
        print("  python test_complete.py --help")
        print("="*80)
    else:
        main()