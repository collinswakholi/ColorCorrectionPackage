"""
Simplified Test Script for NEW ColorCorrectionPipeline (v1.3.0)
================================================================
Tests the color correction pipeline with clear output and metrics.
"""

from pathlib import Path
import numpy as np

# Import NEW pipeline
from ColorCorrectionPipeline import ColorCorrection, Config
from ColorCorrectionPipeline.io import read_image, write_image

# Configuration
WHITE_IMAGE = Path("Test_Images/white.JPG")
SAMPLE_IMAGE = Path("Test_Images/Sample_1.JPG")
OUTPUT_DIR = Path("test_new_pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("COLOR CORRECTION PIPELINE TEST - NEW (v1.3.0)")
print("=" * 80)
print()

# Load images
print("📁 Loading images...")
import cv2
white_img = cv2.imread(str(WHITE_IMAGE))  # BGR uint8 for FFC
sample_img = read_image(str(SAMPLE_IMAGE))  # RGB float64
print(f"   ✓ White image: {white_img.shape}, {white_img.dtype}")
print(f"   ✓ Sample image: {sample_img.shape}, {sample_img.dtype}")
print()

# Configure pipeline
print("⚙️  Configuring pipeline...")
config = Config(
    do_ffc=True,
    do_gc=True,
    do_wb=True,
    do_cc=True,
    save=False,
    check_saturation=False,
    FFC_kwargs={
        'fit_method': 'linear',
        'get_deltaE': True,
        'verbose': False
    },
    GC_kwargs={
        'max_degree': 5,
        'get_deltaE': True,
        'verbose': False
    },
    WB_kwargs={
        'get_deltaE': True
    },
    CC_kwargs={
        'cc_method': 'ours', # "ours", "conv"
        'mtd': 'linear', #
        'degree': 2,
        'max_iterations': 1000,
        'random_state': 42,
        'get_deltaE': True,
        'verbose': False,
        'n_samples': 50
    }
)
print("   ✓ Pipeline configured")
print()

# Run pipeline
print("🚀 Running pipeline...")
print("-" * 80)
pipeline = ColorCorrection()
metrics, images, errors = pipeline.run(
    Image=sample_img,
    config=config,
    White_Image=white_img
)
print("-" * 80)
print()

# Display results
if errors:
    print("❌ Pipeline completed with errors")
    print()
else:
    print("✅ Pipeline completed successfully")
    print()

# Save images
print("💾 Saving output images...")
for stage_name, img in images.items():
    output_path = OUTPUT_DIR / f"{stage_name}.png"
    write_image(str(output_path), img, color_space='rgb')
    print(f"   ✓ {stage_name:5s} → {output_path.name}")
print()

# Display metrics summary
print("=" * 80)
print("📊 METRICS SUMMARY (DeltaE - Lower is Better)")
print("=" * 80)
print()

def find_deltae_mean_key(stage_metrics, position):
    """Return the exact DeltaE mean key for a before/after metric dict."""
    position = position.lower()
    matches = [
        key
        for key in stage_metrics
        if "deltae" in key.lower()
        and position in key.lower()
        and key.lower().endswith("_mean")
    ]
    return matches[0] if matches else None


# Extract DeltaE metrics explicitly. The metrics dict also includes MSE/MAE means,
# so a generic "before/after + mean" search would print the wrong values.
stages = ['_FFC', '_GC', '_WB', '_CC']
print(f"{'Stage':<8} {'Before':<10} {'After':<10} {'Δ Change':<12} {'Quality'}")
print("-" * 80)

for stage in stages:
    if stage in metrics and metrics[stage]:
        m = metrics[stage]
        stage_display = stage[1:]  # Remove leading underscore
        
        before_key = find_deltae_mean_key(m, "before")
        after_key = find_deltae_mean_key(m, "after")
        
        if before_key and after_key:
            before_val = m[before_key]
            after_val = m[after_key]
            change = after_val - before_val
            
            # Quality assessment based on after value
            if after_val < 2.0:
                quality = "🟢 Excellent"
            elif after_val < 5.0:
                quality = "🟡 Good"
            elif after_val < 10.0:
                quality = "🟠 Fair"
            else:
                quality = "🔴 Poor"
            
            # Format change with sign
            change_str = f"{change:+.2f}"
            print(f"{stage_display:<8} {before_val:<10.2f} {after_val:<10.2f} {change_str:>10s}  {quality}")

print()
print("=" * 80)
print("📈 OVERALL PIPELINE PERFORMANCE")
print("=" * 80)
print()

# Calculate overall improvement
if '_FFC' in metrics and '_CC' in metrics:
    ffc_before_key = find_deltae_mean_key(metrics['_FFC'], "before")
    cc_after_key = find_deltae_mean_key(metrics['_CC'], "after")
    ffc_before = metrics['_FFC'][ffc_before_key] if ffc_before_key else None
    cc_after = metrics['_CC'][cc_after_key] if cc_after_key else None
    
    if ffc_before and cc_after:
        overall_improvement = ((ffc_before - cc_after) / ffc_before * 100)
        print(f"Initial DeltaE:  {ffc_before:.2f}")
        print(f"Final DeltaE:    {cc_after:.2f}")
        print(f"Total Reduction: {ffc_before - cc_after:.2f} ({overall_improvement:.1f}% improvement)")
        print()
        
        if cc_after < 2.0:
            print("🎉 Excellent color accuracy achieved!")
        elif cc_after < 5.0:
            print("✅ Good color accuracy achieved!")
        else:
            print("⚠️  Color accuracy could be improved")

print()

# Test predict_image method (applying saved models to new image)
print("=" * 80)
print("🔄 TESTING PREDICT_IMAGE METHOD")
print("=" * 80)
print()
print("Testing the ability to apply saved models to a new image...")
print("(Using the same sample image as a test case)")
print()

try:
    # Use predict_image to apply the trained models
    print("📸 Applying saved models to image...")
    predicted_images = pipeline.predict_image(sample_img, show=False)
    print(f"   ✓ Prediction completed")
    print(f"   ✓ Generated {len(predicted_images)} stage outputs")
    print()
    
    # Save predicted images
    print("💾 Saving predicted images...")
    predict_output_dir = OUTPUT_DIR / "predicted"
    predict_output_dir.mkdir(exist_ok=True)
    
    for stage_name, img in predicted_images.items():
        output_path = predict_output_dir / f"{stage_name}.png"
        write_image(str(output_path), img, color_space='rgb')
        print(f"   ✓ {stage_name:5s} → predicted/{output_path.name}")
    print()
    
    # Compare predicted vs original pipeline output
    print("🔍 Comparing predict_image vs pipeline.run outputs...")
    print()
    
    max_diffs = []
    # Match stage names - predicted uses "FFC", "GC" etc. while images uses "_FFC", "_GC" etc.
    stage_mapping = {
        'FFC': '_FFC',
        'GC': '_GC',
        'WB': '_WB',
        'CC': '_CC'
    }
    
    for pred_stage, orig_stage in stage_mapping.items():
        if pred_stage in predicted_images and orig_stage in images:
            original = images[orig_stage]
            predicted = predicted_images[pred_stage]
            
            diff = np.abs(original - predicted)
            max_diff = diff.max()
            mean_diff = diff.mean()
            max_diffs.append(max_diff)
            
            # Status indicator
            if max_diff < 1e-10:
                status = "✅ Identical"
            elif max_diff < 1e-6:
                status = "✅ Nearly identical"
            elif max_diff < 0.001:
                status = "⚠️  Minor differences"
            else:
                status = "❌ Significant differences"
            
            print(f"   {pred_stage:5s}: max_diff={max_diff:.9f}, mean_diff={mean_diff:.9f} {status}")
    
    print()
    if all(d < 1e-6 for d in max_diffs):
        print("✅ predict_image method works correctly - outputs match pipeline.run!")
    elif all(d < 0.001 for d in max_diffs):
        print("✅ predict_image method works well - minor numerical differences only")
    else:
        print("⚠️  predict_image has some differences from pipeline.run")
    
except Exception as e:
    print(f"❌ predict_image test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print(f"✅ All tests complete! Outputs saved to: {OUTPUT_DIR}")
print("=" * 80)
