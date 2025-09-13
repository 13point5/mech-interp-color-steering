#!/usr/bin/env python3
"""
HTML Color Distribution Analyzer for Steering Experiments

This script analyzes HTML files generated from color steering experiments to quantify
the effectiveness of color interventions by calculating area percentages of different colors.
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
from bs4 import BeautifulSoup
import cssutils
import pandas as pd


@dataclass
class ColorInfo:
    """Information about a color found in HTML"""
    hex_code: str
    color_name: str
    css_property: str  # 'background-color', 'color', 'border-color', etc.
    element_tag: str
    element_class: str
    element_id: str
    area_weight: float  # Estimated area contribution


@dataclass
class ElementMetrics:
    """Metrics for an HTML element"""
    tag: str
    classes: List[str]
    element_id: str
    colors: List[ColorInfo]
    estimated_area: float
    display_type: str  # block, inline, none, etc.


class CSSParser:
    """Enhanced CSS parser to extract color information"""
    
    def __init__(self):
        # Suppress CSS parsing warnings
        cssutils.log.setLevel(40)
        
        # Color properties that affect visual appearance
        self.color_properties = {
            'color', 'background-color', 'background', 'border-color',
            'border-top-color', 'border-right-color', 'border-bottom-color', 'border-left-color',
            'outline-color', 'text-decoration-color', 'box-shadow', 'text-shadow'
        }
    
    def extract_colors_from_css_value(self, css_value: str, property_name: str) -> List[str]:
        """Extract color values from CSS property value"""
        colors = []
        
        # Handle gradients
        if 'gradient' in css_value.lower():
            colors.extend(self._extract_gradient_colors(css_value))
        
        # Handle regular color values
        colors.extend(self._extract_regular_colors(css_value))
        
        return colors
    
    def _extract_gradient_colors(self, gradient_str: str) -> List[str]:
        """Extract colors from CSS gradient strings"""
        colors = []
        
        # Find all color values in gradient (hex, rgb, rgba, hsl, named colors)
        color_patterns = [
            r'#[A-Fa-f0-9]{6}|#[A-Fa-f0-9]{3}',  # Hex colors
            r'rgb\([^)]+\)',  # RGB colors
            r'rgba\([^)]+\)',  # RGBA colors
            r'hsl\([^)]+\)',  # HSL colors
            r'hsla\([^)]+\)',  # HSLA colors
        ]
        
        for pattern in color_patterns:
            colors.extend(re.findall(pattern, gradient_str, re.IGNORECASE))
        
        # Also look for named colors (this is more complex but we'll handle common ones)
        named_color_pattern = r'\b(red|green|blue|yellow|orange|purple|pink|cyan|magenta|lime|navy|maroon|olive|teal|silver|gray|grey|black|white|transparent)\b'
        colors.extend(re.findall(named_color_pattern, gradient_str, re.IGNORECASE))
        
        return colors
    
    def _extract_regular_colors(self, css_value: str) -> List[str]:
        """Extract regular color values (non-gradient)"""
        colors = []
        
        # Skip gradient values as they're handled separately
        if 'gradient' in css_value.lower():
            return colors
        
        # Color patterns
        patterns = [
            r'#[A-Fa-f0-9]{6}|#[A-Fa-f0-9]{3}',  # Hex
            r'rgb\([^)]+\)',  # RGB
            r'rgba\([^)]+\)',  # RGBA
            r'hsl\([^)]+\)',  # HSL
            r'hsla\([^)]+\)',  # HSLA
        ]
        
        for pattern in patterns:
            colors.extend(re.findall(pattern, css_value, re.IGNORECASE))
        
        # Named colors
        named_color_pattern = r'\b(red|green|blue|yellow|orange|purple|pink|cyan|magenta|lime|navy|maroon|olive|teal|silver|gray|grey|black|white)\b'
        colors.extend(re.findall(named_color_pattern, css_value, re.IGNORECASE))
        
        return colors


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


class ColorClassifier:
    """Classifies colors into steering categories"""
    
    def __init__(self):
        self.steering_colors = ['yellow', 'red', 'green', 'blue', 'orange']
        
    def normalize_color_to_hex(self, color_str: str) -> Optional[str]:
        """Convert any color format to hex"""
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
                numbers = re.findall(r'\d+', color_str)
                if len(numbers) >= 3:
                    r, g, b = int(numbers[0]), int(numbers[1]), int(numbers[2])
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
        
        # Special cases
        if color_str == 'transparent':
            return None
        
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


class HTMLColorAnalyzer:
    """Main analyzer for HTML color distribution"""
    
    def __init__(self):
        self.css_parser = CSSParser()
        self.color_classifier = ColorClassifier()
        
        # Default viewport dimensions for area calculations
        self.viewport_width = 1200
        self.viewport_height = 800
    
    def analyze_html_file(self, html_path: str) -> Dict:
        """Analyze a single HTML file for color distribution"""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            return {'error': f"Failed to read file: {e}"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract CSS
        css_rules = self._extract_css_rules(soup)
        
        # Analyze elements
        elements = self._analyze_elements(soup, css_rules)
        
        # Calculate color statistics
        color_stats = self._calculate_color_statistics(elements)
        
        return {
            'file_path': html_path,
            'total_elements': len(elements),
            'color_statistics': color_stats,
            'elements': elements
        }
    
    def _extract_css_rules(self, soup: BeautifulSoup) -> Dict:
        """Extract CSS rules from style tags and link tags"""
        css_rules = defaultdict(dict)
        
        # Extract from <style> tags
        for style_tag in soup.find_all('style'):
            css_text = style_tag.string
            if css_text:
                try:
                    sheet = cssutils.parseString(css_text)
                    for rule in sheet:
                        if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
                            selector = rule.selectorText
                            for prop in rule.style:
                                if prop.name in self.css_parser.color_properties:
                                    colors = self.css_parser.extract_colors_from_css_value(
                                        prop.value, prop.name
                                    )
                                    if colors:
                                        css_rules[selector][prop.name] = colors
                except Exception as e:
                    # If CSS parsing fails, try regex extraction
                    self._extract_css_with_regex(css_text, css_rules)
        
        return css_rules
    
    def _extract_css_with_regex(self, css_text: str, css_rules: Dict):
        """Fallback CSS extraction using regex"""
        # Simple regex to extract CSS rules
        rule_pattern = r'([^{]+)\s*{\s*([^}]+)\s*}'
        
        for match in re.finditer(rule_pattern, css_text):
            selector = match.group(1).strip()
            properties = match.group(2).strip()
            
            # Extract color properties
            for prop_name in self.css_parser.color_properties:
                prop_pattern = rf'{prop_name}\s*:\s*([^;]+)'
                prop_match = re.search(prop_pattern, properties, re.IGNORECASE)
                if prop_match:
                    prop_value = prop_match.group(1).strip()
                    colors = self.css_parser.extract_colors_from_css_value(prop_value, prop_name)
                    if colors:
                        css_rules[selector][prop_name] = colors
    
    def _analyze_elements(self, soup: BeautifulSoup, css_rules: Dict) -> List[ElementMetrics]:
        """Analyze all elements in the HTML"""
        elements = []
        
        for element in soup.find_all():
            if element.name in ['script', 'style', 'meta', 'link', 'title']:
                continue
            
            element_colors = []
            
            # Get inline styles
            style_attr = element.get('style', '')
            if style_attr:
                element_colors.extend(self._extract_colors_from_inline_style(style_attr, element))
            
            # Apply CSS rules
            element_colors.extend(self._apply_css_rules(element, css_rules))
            
            # Estimate area
            estimated_area = self._estimate_element_area(element)
            
            # Get display type
            display_type = self._get_display_type(element, css_rules)
            
            metrics = ElementMetrics(
                tag=element.name,
                classes=element.get('class', []),
                element_id=element.get('id', ''),
                colors=element_colors,
                estimated_area=estimated_area,
                display_type=display_type
            )
            
            elements.append(metrics)
        
        return elements
    
    def _extract_colors_from_inline_style(self, style_attr: str, element) -> List[ColorInfo]:
        """Extract colors from inline style attribute"""
        colors = []
        
        # Parse inline styles
        style_declarations = style_attr.split(';')
        
        for declaration in style_declarations:
            if ':' in declaration:
                prop, value = declaration.split(':', 1)
                prop = prop.strip().lower()
                value = value.strip()
                
                if prop in self.css_parser.color_properties:
                    color_values = self.css_parser.extract_colors_from_css_value(value, prop)
                    
                    for color_val in color_values:
                        hex_color = self.color_classifier.normalize_color_to_hex(color_val)
                        if hex_color:
                            color_name = self.color_classifier.classify_color(hex_color)
                            if color_name:  # Only include colors that are in our steering categories
                                color_info = ColorInfo(
                                    hex_code=hex_color,
                                    color_name=color_name,
                                    css_property=prop,
                                    element_tag=element.name,
                                    element_class=' '.join(element.get('class', [])),
                                    element_id=element.get('id', ''),
                                    area_weight=1.0
                                )
                                colors.append(color_info)
        
        return colors
    
    def _apply_css_rules(self, element, css_rules: Dict) -> List[ColorInfo]:
        """Apply CSS rules to extract colors for an element"""
        colors = []
        
        # Simple selector matching (could be enhanced)
        selectors_to_check = [
            element.name,  # Tag selector
            f"#{element.get('id', '')}" if element.get('id') else None,  # ID selector
        ]
        
        # Class selectors
        for class_name in element.get('class', []):
            selectors_to_check.append(f".{class_name}")
        
        # Check each selector
        for selector in selectors_to_check:
            if selector and selector in css_rules:
                for prop_name, color_values in css_rules[selector].items():
                    for color_val in color_values:
                        hex_color = self.color_classifier.normalize_color_to_hex(color_val)
                        if hex_color:
                            color_name = self.color_classifier.classify_color(hex_color)
                            if color_name:  # Only include colors that are in our steering categories
                                color_info = ColorInfo(
                                    hex_code=hex_color,
                                    color_name=color_name,
                                    css_property=prop_name,
                                    element_tag=element.name,
                                    element_class=' '.join(element.get('class', [])),
                                    element_id=element.get('id', ''),
                                    area_weight=1.0
                                )
                                colors.append(color_info)
        
        return colors
    
    def _estimate_element_area(self, element) -> float:
        """Estimate the visual area of an element"""
        # Simple heuristic based on element type and content
        tag = element.name.lower()
        
        # Base areas for different element types
        area_weights = {
            'body': 1.0,
            'header': 0.8,
            'main': 0.9,
            'section': 0.7,
            'article': 0.6,
            'div': 0.5,
            'nav': 0.3,
            'aside': 0.3,
            'footer': 0.4,
            'h1': 0.4, 'h2': 0.3, 'h3': 0.25, 'h4': 0.2, 'h5': 0.15, 'h6': 0.1,
            'p': 0.2,
            'button': 0.1,
            'a': 0.05,
            'span': 0.02,
            'img': 0.3,
            'table': 0.6,
            'tr': 0.1,
            'td': 0.05,
            'th': 0.05,
        }
        
        base_area = area_weights.get(tag, 0.1)
        
        # Adjust for text content length
        text_content = element.get_text(strip=True) if element else ''
        text_factor = min(len(text_content) / 100.0, 2.0)  # Cap at 2x
        
        # Adjust for child elements
        child_factor = min(len(list(element.children)) / 10.0, 1.5)  # Cap at 1.5x
        
        return base_area * (1 + text_factor * 0.5 + child_factor * 0.3)
    
    def _get_display_type(self, element, css_rules: Dict) -> str:
        """Determine the display type of an element"""
        # Default display types
        block_elements = {
            'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'main', 
            'section', 'article', 'nav', 'aside', 'footer', 'ul', 'ol', 'li', 'table'
        }
        
        inline_elements = {'span', 'a', 'strong', 'em', 'code', 'small'}
        
        if element.name in block_elements:
            return 'block'
        elif element.name in inline_elements:
            return 'inline'
        
        return 'inline'  # Default
    
    def _calculate_color_statistics(self, elements: List[ElementMetrics]) -> Dict:
        """Calculate color distribution statistics"""
        color_areas = defaultdict(float)
        total_area = 0
        color_counts = defaultdict(int)
        
        for element in elements:
            if element.display_type == 'none':
                continue
            
            element_area = element.estimated_area
            total_area += element_area
            
            # If element has no steering colors, skip it for our analysis
            if not element.colors:
                continue
            else:
                # Distribute area among colors (could weight differently for background vs text)
                area_per_color = element_area / len(element.colors)
                
                for color_info in element.colors:
                    color_areas[color_info.color_name] += area_per_color
                    color_counts[color_info.color_name] += 1
        
        # Calculate percentages
        color_percentages = {}
        if total_area > 0:
            for color_name, area in color_areas.items():
                color_percentages[color_name] = (area / total_area) * 100
        
        return {
            'color_areas': dict(color_areas),
            'color_percentages': color_percentages,
            'color_counts': dict(color_counts),
            'total_area': total_area
        }


class ExperimentAnalyzer:
    """Analyze steering experiments across all HTML files"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.analyzer = HTMLColorAnalyzer()
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
    parser = argparse.ArgumentParser(description='Analyze HTML color distribution for steering experiments')
    parser.add_argument('--input-dir', type=str, 
                       default='/home/ubuntu/mech-interp-color-steering/comprehensive_steering_results_act_add',
                       help='Directory containing experiment results')
    parser.add_argument('--output-file', type=str, 
                       default='color_analysis_results.json',
                       help='Output file for analysis results')
    parser.add_argument('--csv-output', type=str,
                       default='color_analysis_summary.csv',
                       help='CSV file for summary statistics')
    
    args = parser.parse_args()
    
    print("Starting HTML Color Analysis...")
    print(f"Input directory: {args.input_dir}")
    
    # Initialize analyzer
    experiment_analyzer = ExperimentAnalyzer(args.input_dir)
    
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
        print("\n=== TOP STEERING CONFIGURATIONS ===")
        top_results = df.nlargest(10, 'average_improvement')
        print(top_results[['experiment', 'steering_color', 'layer', 'strength', 'average_improvement']].to_string(index=False))
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
