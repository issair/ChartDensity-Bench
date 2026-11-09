import os
import random
import json
from PIL import Image
import glob
from datetime import datetime

def create_compound_image(image_paths, output_path, padding=10, background_color='white'):
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)

    n = len(images)
    if n <= 3:
        cols = n
        rows = 1
    elif n <= 6:
        cols = 3
        rows = (n + 2) // 3
    else:
        cols = 3
        rows = 3

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    target_width = min(max_width, 800)
    target_height = int(target_width * max_height / max_width)

    canvas_width = cols * target_width + (cols + 1) * padding
    canvas_height = rows * target_height + (rows + 1) * padding

    canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        row = idx // cols
        col = idx % cols

        x = padding + col * (target_width + padding)
        y = padding + row * (target_height + padding)

        canvas.paste(img_resized, (x, y))

    canvas.save(output_path, 'JPEG', quality=95)

def process_directory(base_dir, group_size):
    test_dir = os.path.join(base_dir, 'test')
    output_dir = os.path.join(base_dir, f'test_compound{group_size}')

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(test_dir, ext)))

    all_images.sort()

    if len(all_images) < group_size:
        print(f"Warning: Number of images ({len(all_images)}) is less than {group_size}, cannot generate compound images with {group_size} images per group")
        return None

    complete_groups = len(all_images) // group_size
    usable_images = complete_groups * group_size
    discarded_images = len(all_images) - usable_images

    print(f"Found {len(all_images)} images")
    print(f"Can generate {complete_groups} compound images with {group_size} images each")
    if discarded_images > 0:
        print(f"Discarding the last {discarded_images} images that cannot be evenly divided")

    shuffled_images = all_images[:usable_images]
    random.shuffle(shuffled_images)

    compound_info = {
        'group_size': group_size,
        'total_images': len(all_images),
        'used_images': usable_images,
        'discarded_images': discarded_images,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'compounds': []
    }

    compound_no = 1
    for i in range(0, usable_images, group_size):
        group_images = shuffled_images[i:i+group_size]

        output_filename = f'{group_size}_{compound_no}.jpg'
        output_path = os.path.join(output_dir, output_filename)

        create_compound_image(group_images, output_path)

        compound_record = {
            'compound_id': compound_no,
            'output_filename': output_filename,
            'source_images': [os.path.basename(img_path) for img_path in group_images],
        }
        compound_info['compounds'].append(compound_record)

        compound_no += 1

    json_filename = f'compound_info_{group_size}.json'
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(compound_info, f, ensure_ascii=False, indent=2)

    print(f"Compound info saved to: {json_path}")
    print(f"Done! Generated {complete_groups} compound images in total\n")

    return compound_info

def main():
    base_directory = "ICPR_filtered"

    test_dir = os.path.join(base_directory, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Directory {test_dir} not found")
        return

    group_sizes = [3, 6, 9]

    random_seed = 42
    random.seed(random_seed)

    print("=" * 60)
    print("Compound Image Generator - Full Concatenation Mode")
    print("=" * 60)
    print(f"Base directory: {os.path.abspath(base_directory)}")
    print(f"Random seed: {random_seed}")
    print("-" * 60)

    all_results = {}
    for group_size in group_sizes:
        print(f"\nProcessing compound images with {group_size} images per group:")
        print("-" * 40)
        result = process_directory(base_directory, group_size)
        if result:
            all_results[f'group_{group_size}'] = result

    print("\n" + "=" * 60)
    print("Generation Statistics Report")
    print("=" * 60)

    for group_size in group_sizes:
        key = f'group_{group_size}'
        if key in all_results:
            result = all_results[key]
            print(f"\n{group_size} images per group:")
            print(f"  - Images used: {result['used_images']}")
            print(f"  - Images discarded: {result['discarded_images']}")
            print(f"  - Compound images generated: {len(result['compounds'])}")

    print("\n✅ All compound images generated successfully!")

    print("\nOutput directories:")
    for group_size in group_sizes:
        output_dir = os.path.join(base_directory, f'test_compound{group_size}')
        json_file = os.path.join(output_dir, f'compound_info_{group_size}.json')
        print(f"  - Images: {output_dir}/")
        print(f"  - Info: {json_file}")

if __name__ == "__main__":
    main()