"""
Test EasyOCR on flowchart images
"""

import easyocr
import cv2
import os
import time

def test_flowcharts():
    # Initialize reader
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Test images directory
    test_dir = "test_images"
    
    # Get all PNG images
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    if not image_files:
        print("No PNG images found in test_images/")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print("="*80)
    
    all_results = []
    
    for filename in sorted(image_files):
        filepath = os.path.join(test_dir, filename)
        
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")
        
        # Load image
        img = cv2.imread(filepath)
        if img is None:
            print(f"❌ Failed to load {filename}")
            continue
        
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Perform OCR
        start_time = time.time()
        results = reader.readtext(filepath, detail=1)
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Text blocks detected: {len(results)}")
        
        if not results:
            print("⚠️  No text detected")
            continue
        
        # Sort by y-coordinate (top to bottom)
        sorted_results = sorted(results, key=lambda x: x[0][0][1])
        
        print(f"\n📝 Extracted Text:\n")
        
        extracted_texts = []
        for i, detection in enumerate(sorted_results, 1):
            bbox, text, confidence = detection
            print(f"  {i:2d}. [{confidence:.3f}] {text}")
            extracted_texts.append(text)
        
        # Calculate statistics
        avg_confidence = sum(r[2] for r in results) / len(results)
        
        print(f"\n📊 Statistics:")
        print(f"  Average confidence: {avg_confidence:.2%}")
        print(f"  Total text blocks: {len(results)}")
        
        # Complete text
        complete_text = ' '.join(extracted_texts)
        print(f"\n💬 Complete text:")
        print(f"  {complete_text[:200]}{'...' if len(complete_text) > 200 else ''}")
        
        all_results.append({
            'filename': filename,
            'blocks': len(results),
            'avg_confidence': avg_confidence,
            'processing_time': processing_time,
            'texts': extracted_texts
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n{result['filename']}:")
        print(f"  Blocks: {result['blocks']}")
        print(f"  Avg Confidence: {result['avg_confidence']:.2%}")
        print(f"  Time: {result['processing_time']:.2f}s")
    
    if all_results:
        avg_blocks = sum(r['blocks'] for r in all_results) / len(all_results)
        avg_conf_overall = sum(r['avg_confidence'] for r in all_results) / len(all_results)
        total_time = sum(r['processing_time'] for r in all_results)
        
        print(f"\n{'='*80}")
        print(f"Overall Average Blocks: {avg_blocks:.1f}")
        print(f"Overall Average Confidence: {avg_conf_overall:.2%}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    test_flowcharts()