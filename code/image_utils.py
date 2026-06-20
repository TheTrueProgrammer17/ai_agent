"""
image_utils.py — Image loading and base64 encoding utilities.
Resizes images to max 512px on longest side before encoding to save tokens.
Includes lightweight image quality analysis.
"""

import base64
import io
import os

try:
    from PIL import Image, ImageStat, ImageFilter
except ImportError:
    Image = None
    ImageStat = None
    ImageFilter = None

def check_image_quality(img) -> list:
    """Check image quality and return a list of risk flags."""
    flags = []
    if ImageStat is None or ImageFilter is None:
        return flags
        
    MIN_SIZE = 100
    BLUR_THRESHOLD = 50.0
    DARKNESS_THRESHOLD = 30
    
    if img.width < MIN_SIZE or img.height < MIN_SIZE:
        flags.append("blurry_image")
        
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = stat.mean[0]
    if mean_brightness < DARKNESS_THRESHOLD:
        flags.append("low_light_or_glare")
        
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_variance = edge_stat.var[0]
    
    if edge_variance < BLUR_THRESHOLD:
        flags.append("blurry_image")
        
    return list(set(flags))


def load_image_as_base64(image_path: str):
    """
    Open image with PIL, resize to max 512px on longest side,
    convert to base64 string. Return None, [] if file missing or corrupt.
    Prints a warning if the image cannot be loaded.
    """
    if Image is None:
        print("[image_utils] WARNING: Pillow not installed, cannot load images.")
        return None, []

    if not os.path.exists(image_path):
        print(f"[image_utils] WARNING: Image not found: {image_path}")
        return None, []

    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (handles PNG with transparency)
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Check image quality
            flags = check_image_quality(img)

            # Resize to max 512px on longest side, maintain aspect ratio
            img.thumbnail((512, 512), Image.LANCZOS)

            # Save to buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode("utf-8")
            return b64, flags
    except Exception as e:
        print(f"[image_utils] WARNING: Could not load image '{image_path}': {e}")
        return None, []


def get_image_id(image_path: str) -> str:
    """
    Return filename without extension.
    Example: "images/test/case_001/img_1.jpg" → "img_1"
    """
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    return name


def load_all_images(image_paths_str: str, base_dir: str = "") -> list:
    """
    image_paths_str is a semicolon-separated string from the CSV.
    Returns list of {"image_id": str, "b64": str or None, "valid": bool, "flags": list}.
    If an image cannot be loaded: valid=False, b64=None, flags=[].
    """
    if not image_paths_str or not image_paths_str.strip():
        return []

    raw_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    result = []

    for raw_path in raw_paths:
        # Try to resolve path: use base_dir prefix if file not found as-is
        resolved = raw_path
        if base_dir and not os.path.exists(raw_path):
            candidate = os.path.join(base_dir, raw_path)
            if os.path.exists(candidate):
                resolved = candidate

        image_id = get_image_id(raw_path)
        b64, flags = load_image_as_base64(resolved)

        result.append({
            "image_id": image_id,
            "b64": b64,
            "valid": b64 is not None,
            "flags": flags,
        })

    return result
