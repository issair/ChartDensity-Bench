import base64
from openai import OpenAI
import os
import json
import argparse
import time
from datetime import datetime

# Official OpenAI client initialization
# Set your API key via environment variable: export OPENAI_API_KEY=""
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

unified_prompt = """
# Chart Data Precise Extraction Instructions

## 1. Basic Identification
- **Chart Type Identification**: Identify the chart type
- **Axis Analysis**:
  - Horizontal/vertical axis labels and units
  - Scale range and intervals (for unit conversion reference only)

## 2. Data Point Extraction Rules
- **Line Charts & Scatter Plots**:
  - Using **actual nodes** on the line as data points
  - When no nodes exist, use evenly spaced points based on scale intervals
  - Baseline: Use the x-value (horizontal axis value) of data points as the baseline
- **Bar Charts**:
  - Use the **center of each bar's top** as the data point
  - Baseline: Use the x-value (horizontal axis value) corresponding to each bar as the baseline
  - For horizontal bar charts: Use y-axis values as baseline
- **Stacked Bar Charts**: Recognize both individual segment values (stacked portions) and total bar height

## 3. Specific Scenario Handling Rules
- **Multi-series Processing**: Distinguish data sequences of different lines/bars
- **Stacked Charts**: Record individual components and totals
- **Dual-axis Charts**: Clearly indicate which axis each series corresponds to
- **Mixed Charts**: Process each chart type according to its respective rules and output in respective tables

## 4. Output Format Specification
| Series Name | X Value | Y Value |
"""

# Image encoding function
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Save error index
def save_error_index(error_list, result_dir):
    """Save the error index as a JSON file"""
    if not error_list:
        return None

    error_path = os.path.join(result_dir, "error.json")
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(error_list, f, indent=4, ensure_ascii=False)

    print(f"📝 Error index file saved: {error_path}")
    return error_path

# Save a single image's MD file
def save_single_md(content, filename, result_dir, recover_mode=False):
    """Save the extraction result of a single image to a separate MD file"""
    base_name = os.path.splitext(filename)[0]
    prefix = "recover_" if recover_mode else ""
    md_filename = f"{prefix}{base_name}.md"
    md_path = os.path.join(result_dir, md_filename)
    
    md_content = f"""# 📊 Chart Data Extraction Result

## Image Information
- **Image File**: {filename}
- **Processing Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Processing Mode**: {"🔧 Recovery Mode" if recover_mode else "🚀 Normal Mode"}

---

## Extraction Result

{content}
"""
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return md_path

# Create summary report
def create_summary_report(success_count, error_files, total_count, mode_label, work_dir, result_dir, recover_mode):
    """Create a processing summary report"""
    report_content = f"""# 📊 Chart Data Extraction Summary Report

## Basic Information
- **Processing Mode**: {mode_label}
- **Generated Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Images Processed**: {total_count}
- **Working Directory**: {work_dir}
- **Result Directory**: {result_dir}

---

## 📈 Processing Statistics
- ✅ Successfully processed: {success_count} images
- ❌ Failed to process: {len(error_files)} images
- 📊 Success rate: {(success_count / total_count * 100) if total_count > 0 else 0:.1f}%

"""

    if error_files:
        report_content += "\n### Failed Image List\n"
        for file in error_files:
            report_content += f"- `{file}`\n"
        
        report_content += f"\n> **Error Index File**: `error.json`"

    if not recover_mode and error_files:
        report_content += f"\n> 💡 Tip: Error entries found. You can run recovery mode to process these files: \n> Command: `python script.py {work_dir} --recover`"

    return report_content

# Main processing function
def process_images(image_list, image_dir, work_dir, recover_mode=False):
    """Process the image list, supporting recovery mode"""
    mode_label = "🔧 Recovery Mode" if recover_mode else "🚀 Normal Mode"

    result_dir = os.path.join(work_dir, "results_gemini")
    os.makedirs(result_dir, exist_ok=True)

    success_count = 0
    error_files = []
    error_list = []

    for idx, image_info in enumerate(image_list):
        if isinstance(image_info, dict):
            filename = image_info.get("filename", "")
        else:
            filename = image_info
            
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            error_msg = f"⚠ Image does not exist: {image_path}"
            print(error_msg)
            error_files.append(filename)
            error_list.append({
                "filename": filename,
                "error": "file not found",
                "timestamp": datetime.now().isoformat()
            })
            continue

        print(f"\n{'=' * 60}")
        print(f"🔍 Processing image [{idx + 1}/{len(image_list)}]: {filename}")
        print(f"{'=' * 60}")

        try:
            image_base64 = encode_image_to_base64(image_path)

            response = client.chat.completions.create(
                model="",  # Choose your model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": unified_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
            )

            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                print(f"\n{'=' * 60}")
                print(f"📝 Raw Output:")
                print(f"{'=' * 60}")
                print(content)
                print(f"{'=' * 60}\n")
                
                if '|' in content:
                    pipe_index = content.find('|')
                    extracted_content = content[pipe_index:].strip()
                else:
                    extracted_content = "No table returned"
                
                success_count += 1
                md_path = save_single_md(extracted_content, filename, result_dir, recover_mode)

                preview = extracted_content[:50] + "..." if len(extracted_content) > 50 else extracted_content
                print(f"✅ Successfully processed: {filename}")
                print(f"📄 Content preview:\n{preview}\n")
                print(f"💾 Saved to: {md_path}")
                
                time.sleep(1)

            else:
                error_msg = f"❌ No valid response received: {filename}"
                print(error_msg)
                error_files.append(filename)
                error_list.append({
                    "filename": filename,
                    "error": "no valid response",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            error_msg = f"❌ Error while processing image({filename}): {str(e)}"
            print(error_msg)
            error_files.append(filename)
            error_list.append({
                "filename": filename,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    if error_files:
        save_error_index(error_list, result_dir)

    report_content = create_summary_report(success_count, error_files, len(image_list), mode_label, work_dir, result_dir, recover_mode)
    
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "Recovery" if recover_mode else ""
    work_dir_basename = os.path.basename(work_dir)
    report_filename = os.path.join(result_dir, f"Processing_{prefix}Report_{work_dir_basename}_{report_timestamp}.md")
    
    with open(report_filename, 'w', encoding='utf-8') as report_file:
        report_file.write(report_content)

    print(f"\n{'=' * 60}")
    print(f"✅ Processing complete! Summary report saved to: {report_filename}")
    print(f"📁 All results saved in: {result_dir}")
    print(f"- Success: {success_count} | Failed: {len(error_files)}")
    print(f"{'=' * 60}")

    return report_filename

# Function to process a single directory
def process_directory(work_dir, recover_mode):
    """Process a single directory"""
    print(f"\n{'=' * 60}")
    print(f"{'🔧 Recovery Mode' if recover_mode else '🚀 Normal Mode'} - Processing directory: {work_dir}")
    print(f"{'=' * 60}")

    result_dir = os.path.join(work_dir, "results_gemini")
    os.makedirs(result_dir, exist_ok=True)
    
    error_json_path = os.path.join(result_dir, "error.json")
    
    if recover_mode:
        if not os.path.exists(error_json_path):
            print(f"❌ Error: Recovery mode requires an error.json file, but {error_json_path} was not found")
            return None
            
        try:
            with open(error_json_path, 'r', encoding='utf-8') as f:
                error_data = json.load(f)
            image_list = [item["filename"] for item in error_data if "filename" in item]
            if not image_list:
                print(f"❌ Error: No valid filenames in error.json")
                return None
        except Exception as e:
            print(f"❌ Failed to read error.json: {str(e)}")
            return None
    else:
        if not os.path.exists(work_dir):
            print(f"❌ Error: Directory does not exist {work_dir}")
            return None
            
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        image_list = []
        for file in os.listdir(work_dir):
            file_path = os.path.join(work_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                image_list.append(file)
        
        if not image_list:
            print(f"❌ Error: No image files found in directory {work_dir}")
            return None

    return process_images(image_list, work_dir, work_dir, recover_mode)

def main():
    parser = argparse.ArgumentParser(description='Process image extraction tasks')
    parser.add_argument('directories', type=str, nargs='+', help='Working directory paths (multiple allowed)')
    parser.add_argument('--recover', action='store_true', help='Use recovery mode')
    args = parser.parse_args()

    RECOVER_MODE = args.recover
    DIRECTORIES = args.directories
    
    for work_dir in DIRECTORIES:
        abs_dir = os.path.abspath(work_dir)
        
        try:
            report_path = process_directory(abs_dir, RECOVER_MODE)
            if report_path:
                print(f"✅ Processing complete: {abs_dir}, report path: {report_path}")
            else:
                print(f"❌ Processing failed: {abs_dir}")
        except Exception as e:
            print(f"❌ Error while processing directory {abs_dir}: {str(e)}")

if __name__ == "__main__":
    main()