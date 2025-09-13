#!/usr/bin/env python3
"""
HTML Color Distribution Analyzer using Browser Rendering

This script renders HTML files in a headless browser to get actual calculated styles
and element dimensions, then calculates accurate color area percentages.
"""

import os
import re
import json
import colorsys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import argparse

import webcolors
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def extract_hex_codes(text):
    """
    Extracts hex codes (3 or 6 characters) from a text, including the '#' prefix.
    """
    return re.findall(r'(#[A-Fa-f0-9]{6}|#[A-Fa-f0-9]{3})', text)

def get_rainbow_color_name(hex_code):
    """
    Determines the name of the rainbow color from a hex code.

    Args:
        hex_code (str): The hex code, e.g., '#FF0000'.

    Returns:
        str: The name of the nearest rainbow color, or None if the input is invalid.
    """
    try:
        # Convert hex to RGB tuple
        rgb_tuple = webcolors.hex_to_rgb(hex_code)
    except ValueError:
        return None

    # Convert RGB to HSL. Note: colorsys returns (hue, lightness, saturation).
    r, g, b = [c / 255.0 for c in rgb_tuple]
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # --- FIX: Check for desaturated colors (black, white, gray) first. ---
    # The hue of a desaturated color is meaningless, so we handle these separately.
    # Increased saturation threshold to exclude very light/desaturated colors
    if s < 0.4:  # Low saturation indicates a shade of gray
        if l > 0.85:  # Very light colors -> White
            return "White"
        elif l < 0.3:  # Dark colors (including dark grays) -> Black  
            return "Black"
        else:
            return "Gray"
            
    # --- Now check for specific rainbow colors based on hue ---
    hue_degrees = h * 360

    if 330 <= hue_degrees or hue_degrees < 15:
        return "Red"
    elif 15 <= hue_degrees < 45:
        return "Orange"
    elif 45 <= hue_degrees < 75:
        return "Yellow"
    elif 75 <= hue_degrees < 165:
        return "Green"
    elif 165 <= hue_degrees < 255:
        return "Blue"
    elif 255 <= hue_degrees < 270:
        return "Indigo"
    elif 270 <= hue_degrees < 330:
        return "Violet"

    return None


@dataclass
class ElementColorInfo:
    """Information about a rendered element's colors and dimensions"""
    tag: str
    element_id: str
    classes: List[str]
    x: float
    y: float
    width: float
    height: float
    area: float
    background_color: Optional[str]
    text_color: Optional[str]
    border_colors: List[str]
    is_visible: bool


class ColorClassifier:
    """Classifies colors into steering categories"""
    
    def __init__(self):
        self.steering_colors = ['yellow', 'red', 'green', 'blue', 'orange']
        
    def normalize_color_to_hex(self, color_str: str) -> Optional[str]:
        """Convert any color format to hex"""
        if not color_str or color_str.lower() in ['transparent', 'rgba(0, 0, 0, 0)', 'none']:
            return None
            
        color_str = color_str.strip().lower()
        
        # Already hex
        if color_str.startswith('#'):
            # Normalize 3-digit hex to 6-digit
            if len(color_str) == 4:
                return '#' + ''.join([c*2 for c in color_str[1:]])
            return color_str.upper()
        
        # Handle RGB/RGBA
        if color_str.startswith(('rgb(', 'rgba(')):
            try:
                # Extract numbers from rgb(r,g,b) or rgba(r,g,b,a)
                numbers = re.findall(r'(\d+(?:\.\d+)?)', color_str)
                if len(numbers) >= 3:
                    r, g, b = int(float(numbers[0])), int(float(numbers[1])), int(float(numbers[2]))
                    # Check for transparency
                    if len(numbers) >= 4 and float(numbers[3]) == 0:
                        return None
                    return f"#{r:02X}{g:02X}{b:02X}"
            except:
                pass
        
        # Handle HSL/HSLA (convert to RGB first)
        if color_str.startswith(('hsl(', 'hsla(')):
            try:
                # Extract numbers from hsl(h,s%,l%) or hsla(h,s%,l%,a)
                matches = re.findall(r'(\d+(?:\.\d+)?)', color_str)
                if len(matches) >= 3:
                    h = float(matches[0]) / 360.0
                    s = float(matches[1]) / 100.0
                    l = float(matches[2]) / 100.0
                    # Check for transparency
                    if len(matches) >= 4 and float(matches[3]) == 0:
                        return None
                    r, g, b = colorsys.hls_to_rgb(h, l, s)
                    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
            except:
                pass
        
        # Handle named colors
        try:
            rgb = webcolors.name_to_rgb(color_str)
            return f"#{rgb.red:02X}{rgb.green:02X}{rgb.blue:02X}"
        except ValueError:
            pass
        
        return None
    
    def classify_color(self, hex_color: str) -> Optional[str]:
        """Classify a hex color using the rainbow color function"""
        if not hex_color or hex_color is None:
            return None
        
        # Use the existing rainbow color classification
        color_name = get_rainbow_color_name(hex_color)
        
        # Only return colors that are in our steering categories
        if color_name and color_name.lower() in [c.lower() for c in self.steering_colors]:
            return color_name.lower()
        
        return None


class BrowserColorAnalyzer:
    """Main analyzer using browser rendering for accurate measurements"""
    
    def __init__(self, viewport_width=1200, viewport_height=800):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.color_classifier = ColorClassifier()
        self.driver = None
        
    def _setup_driver(self):
        """Setup headless Chrome driver"""
        if self.driver:
            return
            
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument(f"--window-size={self.viewport_width},{self.viewport_height}")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_window_size(self.viewport_width, self.viewport_height)
    
    def _cleanup_driver(self):
        """Cleanup browser driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def analyze_html_file(self, html_path: str) -> Dict:
        """Analyze a single HTML file using browser rendering"""
        try:
            self._setup_driver()
            
            # Load the HTML file
            file_url = f"file://{os.path.abspath(html_path)}"
            self.driver.get(file_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Get all elements
            elements = self.driver.find_elements(By.XPATH, "//*")
            
            element_infos = []
            total_colored_area = 0
            color_areas = defaultdict(float)
            color_counts = defaultdict(int)
            
            for element in elements:
                try:
                    # Get element info
                    tag = element.tag_name
                    element_id = element.get_attribute('id') or ''
                    classes = (element.get_attribute('class') or '').split()
                    
                    # Get computed styles
                    location = element.location
                    size = element.size
                    
                    # Skip elements with no size or off-screen
                    if size['width'] <= 0 or size['height'] <= 0:
                        continue
                    
                    area = size['width'] * size['height']
                    
                    # Get computed styles
                    background_color = self.driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).backgroundColor", element
                    )
                    text_color = self.driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).color", element
                    )
                    border_top_color = self.driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).borderTopColor", element
                    )
                    border_right_color = self.driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).borderRightColor", element
                    )
                    border_bottom_color = self.driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).borderBottomColor", element
                    )
                    border_left_color = self.driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).borderLeftColor", element
                    )
                    
                    # Check if element is visible
                    is_visible = element.is_displayed()
                    
                    if not is_visible:
                        continue
                    
                    # Normalize colors
                    bg_hex = self.color_classifier.normalize_color_to_hex(background_color)
                    text_hex = self.color_classifier.normalize_color_to_hex(text_color)
                    border_colors = []
                    for border_color in [border_top_color, border_right_color, border_bottom_color, border_left_color]:
                        border_hex = self.color_classifier.normalize_color_to_hex(border_color)
                        if border_hex:
                            border_colors.append(border_hex)
                    
                    # Classify colors
                    bg_class = self.color_classifier.classify_color(bg_hex) if bg_hex else None
                    text_class = self.color_classifier.classify_color(text_hex) if text_hex else None
                    border_classes = [self.color_classifier.classify_color(bc) for bc in border_colors]
                    border_classes = [bc for bc in border_classes if bc]
                    
                    # Create element info
                    element_info = ElementColorInfo(
                        tag=tag,
                        element_id=element_id,
                        classes=classes,
                        x=location['x'],
                        y=location['y'],
                        width=size['width'],
                        height=size['height'],
                        area=area,
                        background_color=bg_hex,
                        text_color=text_hex,
                        border_colors=border_colors,
                        is_visible=is_visible
                    )
                    
                    element_infos.append(element_info)
                    
                    # Count colors and areas
                    has_steering_color = False
                    
                    # Background color gets full area weight
                    if bg_class:
                        color_areas[bg_class] += area
                        color_counts[bg_class] += 1
                        has_steering_color = True
                    
                    # Text color gets smaller weight (10% of area)
                    if text_class and text_class != bg_class:
                        color_areas[text_class] += area * 0.1
                        color_counts[text_class] += 1
                        has_steering_color = True
                    
                    # Border colors get small weight (5% of area each)
                    for border_class in border_classes:
                        if border_class and border_class not in [bg_class, text_class]:
                            color_areas[border_class] += area * 0.05
                            color_counts[border_class] += 1
                            has_steering_color = True
                    
                    if has_steering_color:
                        total_colored_area += area
                        
                except Exception as e:
                    # Skip problematic elements
                    continue
            
            # Calculate percentages
            color_percentages = {}
            if total_colored_area > 0:
                for color_name, area in color_areas.items():
                    color_percentages[color_name] = (area / total_colored_area) * 100
            
            return {
                'file_path': html_path,
                'viewport_size': (self.viewport_width, self.viewport_height),
                'total_elements': len(element_infos),
                'total_colored_area': total_colored_area,
                'color_statistics': {
                    'color_areas': dict(color_areas),
                    'color_percentages': color_percentages,
                    'color_counts': dict(color_counts),
                },
                'elements': element_infos
            }
            
        except Exception as e:
            return {'error': f"Failed to analyze file: {e}"}
        finally:
            self._cleanup_driver()


class ExperimentAnalyzer:
    """Analyze steering experiments across all HTML files using browser rendering"""
    
    def __init__(self, base_dir: str, viewport_width=1200, viewport_height=800):
        self.base_dir = Path(base_dir)
        self.analyzer = BrowserColorAnalyzer(viewport_width, viewport_height)
        self.steering_colors = ['yellow', 'red', 'green', 'blue', 'orange']
    
    def analyze_all_experiments(self) -> Dict:
        """Analyze all experiments in the base directory"""
        results = {
            'experiments': {},
            'baseline': {},
            'summary': {}
        }
        
        # Analyze baseline
        baseline_dir = self.base_dir / 'baseline'
        if baseline_dir.exists():
            print("Analyzing baseline...")
            results['baseline'] = self._analyze_directory(baseline_dir)
        
        # Analyze each experiment
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != 'baseline' and not exp_dir.name.endswith('.json'):
                print(f"Analyzing experiment: {exp_dir.name}")
                results['experiments'][exp_dir.name] = self._analyze_experiment_directory(exp_dir)
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _analyze_directory(self, directory: Path) -> Dict:
        """Analyze all HTML files in a directory"""
        files_results = {}
        
        for html_file in directory.glob('*.html'):
            print(f"  Analyzing: {html_file.name}")
            result = self.analyzer.analyze_html_file(str(html_file))
            files_results[html_file.name] = result
        
        return files_results
    
    def _analyze_experiment_directory(self, exp_dir: Path) -> Dict:
        """Analyze an experiment directory with layer/strength structure"""
        experiment_results = {}
        
        for layer_dir in exp_dir.iterdir():
            if layer_dir.is_dir() and layer_dir.name.startswith('layer_'):
                layer_num = layer_dir.name.replace('layer_', '')
                experiment_results[layer_num] = {}
                
                for strength_dir in layer_dir.iterdir():
                    if strength_dir.is_dir() and strength_dir.name.startswith('strength_'):
                        strength_num = strength_dir.name.replace('strength_', '')
                        print(f"    Layer {layer_num}, Strength {strength_num}")
                        experiment_results[layer_num][strength_num] = self._analyze_directory(strength_dir)
        
        return experiment_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics across all experiments"""
        summary = {
            'steering_effectiveness': {},
            'best_configurations': {},
            'color_trends': {}
        }
        
        baseline_colors = self._extract_baseline_colors(results['baseline'])
        
        for exp_name, exp_data in results['experiments'].items():
            steering_color = self._extract_steering_color(exp_name)
            if steering_color:
                effectiveness = self._calculate_steering_effectiveness(
                    exp_data, baseline_colors, steering_color
                )
                summary['steering_effectiveness'][exp_name] = effectiveness
        
        return summary
    
    def _extract_baseline_colors(self, baseline_data: Dict) -> Dict:
        """Extract color percentages from baseline data"""
        baseline_colors = defaultdict(list)
        
        for file_name, file_data in baseline_data.items():
            if 'color_statistics' in file_data:
                color_percentages = file_data['color_statistics'].get('color_percentages', {})
                for color, percentage in color_percentages.items():
                    baseline_colors[color].append(percentage)
        
        # Average percentages across files
        avg_baseline_colors = {}
        for color, percentages in baseline_colors.items():
            avg_baseline_colors[color] = sum(percentages) / len(percentages) if percentages else 0
        
        return avg_baseline_colors
    
    def _extract_steering_color(self, experiment_name: str) -> Optional[str]:
        """Extract the steering color from experiment name"""
        for color in self.steering_colors:
            if color in experiment_name.lower():
                return color
        return None
    
    def _calculate_steering_effectiveness(self, exp_data: Dict, baseline_colors: Dict, steering_color: str) -> Dict:
        """Calculate how effective the steering was for a specific color"""
        effectiveness_data = {}
        
        baseline_percentage = baseline_colors.get(steering_color, 0)
        
        for layer, layer_data in exp_data.items():
            effectiveness_data[layer] = {}
            
            for strength, strength_data in layer_data.items():
                effectiveness_scores = []
                
                for file_name, file_data in strength_data.items():
                    if 'color_statistics' in file_data:
                        color_percentages = file_data['color_statistics'].get('color_percentages', {})
                        steered_percentage = color_percentages.get(steering_color, 0)
                        
                        # Calculate improvement over baseline
                        improvement = steered_percentage - baseline_percentage
                        effectiveness_scores.append(improvement)
                
                if effectiveness_scores:
                    avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
                    effectiveness_data[layer][strength] = {
                        'average_improvement': avg_effectiveness,
                        'baseline_percentage': baseline_percentage,
                        'scores': effectiveness_scores
                    }
        
        return effectiveness_data


def main():
    parser = argparse.ArgumentParser(description='Analyze HTML color distribution using browser rendering')
    parser.add_argument('--input-dir', type=str, 
                       default='/home/ubuntu/mech-interp-color-steering/comprehensive_steering_results_act_add',
                       help='Directory containing experiment results')
    parser.add_argument('--output-file', type=str, 
                       default='browser_analysis_results.json',
                       help='Output file for analysis results')
    parser.add_argument('--csv-output', type=str,
                       default='browser_analysis_summary.csv',
                       help='CSV file for summary statistics')
    parser.add_argument('--viewport-width', type=int, default=1200,
                       help='Browser viewport width')
    parser.add_argument('--viewport-height', type=int, default=800,
                       help='Browser viewport height')
    
    args = parser.parse_args()
    
    print("Starting Browser-Based HTML Color Analysis...")
    print(f"Input directory: {args.input_dir}")
    print(f"Viewport size: {args.viewport_width}x{args.viewport_height}")
    
    # Initialize analyzer
    experiment_analyzer = ExperimentAnalyzer(
        args.input_dir, 
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height
    )
    
    # Run analysis
    results = experiment_analyzer.analyze_all_experiments()
    
    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {output_path}")
    
    # Create CSV summary
    csv_data = []
    
    # Process steering effectiveness data
    if 'steering_effectiveness' in results['summary']:
        for exp_name, exp_effectiveness in results['summary']['steering_effectiveness'].items():
            steering_color = experiment_analyzer._extract_steering_color(exp_name)
            
            for layer, layer_data in exp_effectiveness.items():
                for strength, strength_data in layer_data.items():
                    csv_data.append({
                        'experiment': exp_name,
                        'steering_color': steering_color,
                        'layer': layer,
                        'strength': strength,
                        'average_improvement': strength_data['average_improvement'],
                        'baseline_percentage': strength_data['baseline_percentage'],
                        'num_files': len(strength_data['scores'])
                    })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(args.csv_output, index=False)
        print(f"Summary CSV saved to: {args.csv_output}")
        
        # Print top results
        print("\n=== TOP STEERING CONFIGURATIONS (Browser-Based) ===")
        top_results = df.nlargest(10, 'average_improvement')
        print(top_results[['experiment', 'steering_color', 'layer', 'strength', 'average_improvement']].to_string(index=False))
    
    print("\nBrowser-based analysis complete!")


if __name__ == "__main__":
    main()
