#!/usr/bin/env python3
"""
Analyze and cluster HTML files from color steering experiments
to find similar outputs across different color experiments.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import hashlib
from typing import Dict, List, Tuple, Set
import difflib

class HTMLAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.html_files = []
        self.file_features = {}
        self.clusters = defaultdict(list)
        
    def find_html_files(self):
        """Find all HTML files in the directory structure"""
        print("Finding HTML files...")
        for html_file in self.base_dir.rglob("*.html"):
            if html_file.name != "comprehensive_results.json":
                self.html_files.append(html_file)
        print(f"Found {len(self.html_files)} HTML files")
        
    def extract_key_features(self, html_content: str) -> Dict:
        """Extract key features from HTML content for comparison"""
        features = {}
        
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
        features['title'] = title_match.group(1) if title_match else ""
        
        # Extract main heading (h1)
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE | re.DOTALL)
        features['main_heading'] = h1_match.group(1) if h1_match else ""
        
        # Extract hero section h2
        hero_h2_matches = re.findall(r'<h2[^>]*>(.*?)</h2>', html_content, re.IGNORECASE | re.DOTALL)
        features['hero_headings'] = hero_h2_matches
        
        # Extract feature headings (h3)
        h3_matches = re.findall(r'<h3[^>]*>(.*?)</h3>', html_content, re.IGNORECASE | re.DOTALL)
        features['feature_headings'] = h3_matches
        
        # Extract CSS colors (hex, rgb, color names)
        color_patterns = [
            r'#[0-9a-fA-F]{3,6}',  # hex colors
            r'rgb\([^)]+\)',        # rgb colors
            r'color:\s*([a-z]+)',   # color names
            r'background[^:]*:\s*([^;]+)',  # background colors
        ]
        
        colors = set()
        for pattern in color_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            colors.update(matches)
        features['colors'] = sorted(list(colors))
        
        # Extract structural elements count
        features['div_count'] = len(re.findall(r'<div', html_content, re.IGNORECASE))
        features['section_count'] = len(re.findall(r'<section', html_content, re.IGNORECASE))
        features['nav_count'] = len(re.findall(r'<nav', html_content, re.IGNORECASE))
        features['button_count'] = len(re.findall(r'<button', html_content, re.IGNORECASE))
        
        # Extract text content (no HTML tags)
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        features['text_content'] = text_content
        features['text_length'] = len(text_content)
        
        # Extract key phrases from text
        words = text_content.lower().split()
        features['key_words'] = [w for w in words if len(w) > 3 and w.isalpha()][:20]  # First 20 significant words
        
        return features
        
    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity score between two feature sets"""
        score = 0.0
        total_weight = 0.0
        
        # Title similarity (weight: 20%)
        title_sim = difflib.SequenceMatcher(None, features1['title'], features2['title']).ratio()
        score += title_sim * 0.2
        total_weight += 0.2
        
        # Main heading similarity (weight: 15%)
        heading_sim = difflib.SequenceMatcher(None, features1['main_heading'], features2['main_heading']).ratio()
        score += heading_sim * 0.15
        total_weight += 0.15
        
        # Feature headings similarity (weight: 25%)
        h3_1 = set(features1['feature_headings'])
        h3_2 = set(features2['feature_headings'])
        if h3_1 or h3_2:
            h3_sim = len(h3_1.intersection(h3_2)) / max(len(h3_1.union(h3_2)), 1)
        else:
            h3_sim = 1.0
        score += h3_sim * 0.25
        total_weight += 0.25
        
        # Color scheme similarity (weight: 20%)
        colors_1 = set(features1['colors'])
        colors_2 = set(features2['colors'])
        if colors_1 or colors_2:
            color_sim = len(colors_1.intersection(colors_2)) / max(len(colors_1.union(colors_2)), 1)
        else:
            color_sim = 1.0
        score += color_sim * 0.2
        total_weight += 0.2
        
        # Structural similarity (weight: 10%)
        struct_keys = ['div_count', 'section_count', 'nav_count', 'button_count']
        struct_sim = 0
        for key in struct_keys:
            val1, val2 = features1[key], features2[key]
            if val1 == val2:
                struct_sim += 1
            elif val1 > 0 and val2 > 0:
                struct_sim += 1 - abs(val1 - val2) / max(val1, val2)
        struct_sim /= len(struct_keys)
        score += struct_sim * 0.1
        total_weight += 0.1
        
        # Text content similarity (weight: 10%)
        text_sim = difflib.SequenceMatcher(None, features1['text_content'], features2['text_content']).ratio()
        score += text_sim * 0.1
        total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0
        
    def parse_file_path(self, file_path: Path) -> Dict:
        """Parse experiment info from file path"""
        parts = file_path.parts
        
        info = {
            'full_path': str(file_path),
            'filename': file_path.name,
            'experiment': None,
            'layer': None,
            'strength': None,
            'prompt': None
        }
        
        # Extract experiment name (color_neutral, baseline, etc.)
        for part in parts:
            if any(color in part for color in ['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'purple']):
                info['experiment'] = part
                break
            elif part == 'baseline':
                info['experiment'] = 'baseline'
                break
                
        # Extract layer, strength, prompt from path
        for part in parts:
            if part.startswith('layer_'):
                try:
                    info['layer'] = int(part.split('_')[1])
                except:
                    pass
            elif part.startswith('strength_'):
                try:
                    info['strength'] = int(part.split('_')[1])
                except:
                    pass
                    
        # Extract prompt from filename
        if 'prompt_' in file_path.name:
            try:
                prompt_part = file_path.name.split('prompt_')[1].split('_')[0]
                info['prompt'] = int(prompt_part)
            except:
                pass
                
        return info
        
    def analyze_all_files(self):
        """Analyze all HTML files and extract features"""
        print("Analyzing HTML files...")
        
        for i, html_file in enumerate(self.html_files):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(self.html_files)} files")
                
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                features = self.extract_key_features(content)
                path_info = self.parse_file_path(html_file)
                
                self.file_features[str(html_file)] = {
                    'features': features,
                    'path_info': path_info,
                    'content_hash': hashlib.md5(content.encode()).hexdigest()
                }
                
            except Exception as e:
                print(f"Error processing {html_file}: {e}")
                
        print(f"Analyzed {len(self.file_features)} files")
        
    def find_clusters(self, similarity_threshold: float = 0.8):
        """Find clusters of similar files"""
        print(f"Finding clusters with similarity threshold {similarity_threshold}...")
        
        file_paths = list(self.file_features.keys())
        clustered = set()
        cluster_id = 0
        
        for i, file1 in enumerate(file_paths):
            if file1 in clustered:
                continue
                
            current_cluster = [file1]
            clustered.add(file1)
            
            for j, file2 in enumerate(file_paths[i+1:], i+1):
                if file2 in clustered:
                    continue
                    
                similarity = self.calculate_similarity(
                    self.file_features[file1]['features'],
                    self.file_features[file2]['features']
                )
                
                if similarity >= similarity_threshold:
                    current_cluster.append(file2)
                    clustered.add(file2)
                    
            if len(current_cluster) > 1:
                self.clusters[cluster_id] = current_cluster
                cluster_id += 1
                
        print(f"Found {len(self.clusters)} clusters")
        
    def find_cross_experiment_clusters(self, similarity_threshold: float = 0.7):
        """Find clusters specifically across different color experiments"""
        print(f"Finding cross-experiment clusters...")
        
        cross_clusters = {}
        cluster_id = 0
        
        file_paths = list(self.file_features.keys())
        
        for i, file1 in enumerate(file_paths):
            exp1 = self.file_features[file1]['path_info']['experiment']
            if not exp1:
                continue
                
            cluster_files = [file1]
            
            for j, file2 in enumerate(file_paths):
                if i == j:
                    continue
                    
                exp2 = self.file_features[file2]['path_info']['experiment']
                if not exp2 or exp1 == exp2:  # Skip same experiment
                    continue
                    
                similarity = self.calculate_similarity(
                    self.file_features[file1]['features'],
                    self.file_features[file2]['features']
                )
                
                if similarity >= similarity_threshold:
                    cluster_files.append(file2)
                    
            if len(cluster_files) > 1:
                # Check if this cluster already exists (different starting file)
                experiments_in_cluster = set(self.file_features[f]['path_info']['experiment'] for f in cluster_files)
                
                is_duplicate = False
                for existing_cluster in cross_clusters.values():
                    existing_experiments = set(self.file_features[f]['path_info']['experiment'] for f in existing_cluster)
                    if experiments_in_cluster == existing_experiments:
                        # Check if files overlap significantly
                        overlap = len(set(cluster_files).intersection(set(existing_cluster)))
                        if overlap > len(cluster_files) * 0.5:
                            is_duplicate = True
                            break
                            
                if not is_duplicate:
                    cross_clusters[cluster_id] = cluster_files
                    cluster_id += 1
                    
        return cross_clusters
        
    def print_cluster_summary(self, clusters: Dict):
        """Print a summary of clusters"""
        for cluster_id, files in clusters.items():
            print(f"\n=== CLUSTER {cluster_id} ({len(files)} files) ===")
            
            # Group by experiment
            by_experiment = defaultdict(list)
            for file_path in files:
                exp = self.file_features[file_path]['path_info']['experiment']
                by_experiment[exp].append(file_path)
                
            for exp, exp_files in by_experiment.items():
                print(f"  {exp}: {len(exp_files)} files")
                for file_path in exp_files[:3]:  # Show first 3 files
                    path_info = self.file_features[file_path]['path_info']
                    print(f"    - Layer {path_info['layer']}, Strength {path_info['strength']}, Prompt {path_info['prompt']}")
                if len(exp_files) > 3:
                    print(f"    ... and {len(exp_files) - 3} more")
                    
            # Show sample features
            sample_file = files[0]
            features = self.file_features[sample_file]['features']
            print(f"  Sample features:")
            print(f"    Title: {features['title'][:50]}...")
            print(f"    Main heading: {features['main_heading'][:50]}...")
            print(f"    Feature headings: {features['feature_headings'][:3]}")
            print(f"    Colors: {features['colors'][:5]}")
            
    def save_results(self, output_file: str):
        """Save clustering results to JSON"""
        results = {
            'total_files': len(self.file_features),
            'clusters': {},
            'cross_experiment_clusters': {},
            'file_details': {}
        }
        
        # Regular clusters
        for cluster_id, files in self.clusters.items():
            results['clusters'][str(cluster_id)] = {
                'files': files,
                'experiments': list(set(self.file_features[f]['path_info']['experiment'] for f in files)),
                'size': len(files)
            }
            
        # Cross-experiment clusters
        cross_clusters = self.find_cross_experiment_clusters()
        for cluster_id, files in cross_clusters.items():
            results['cross_experiment_clusters'][str(cluster_id)] = {
                'files': files,
                'experiments': list(set(self.file_features[f]['path_info']['experiment'] for f in files)),
                'size': len(files)
            }
            
        # File details
        for file_path, data in self.file_features.items():
            results['file_details'][file_path] = {
                'experiment': data['path_info']['experiment'],
                'layer': data['path_info']['layer'],
                'strength': data['path_info']['strength'],
                'prompt': data['path_info']['prompt'],
                'title': data['features']['title'],
                'main_heading': data['features']['main_heading'],
                'feature_headings': data['features']['feature_headings'],
                'colors': data['features']['colors'][:10],  # Limit colors
                'content_hash': data['content_hash']
            }
            
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {output_file}")
        
        return cross_clusters

def main():
    base_dir = "/home/ubuntu/mech-interp-color-steering/comprehensive_steering_results_act_add"
    analyzer = HTMLAnalyzer(base_dir)
    
    # Analyze all files
    analyzer.find_html_files()
    analyzer.analyze_all_files()
    
    # Find regular clusters
    analyzer.find_clusters(similarity_threshold=0.8)
    
    # Find cross-experiment clusters
    cross_clusters = analyzer.find_cross_experiment_clusters(similarity_threshold=0.7)
    
    # Print summaries
    print("\n" + "="*60)
    print("CROSS-EXPERIMENT CLUSTERS (Similar files across different colors)")
    print("="*60)
    analyzer.print_cluster_summary(cross_clusters)
    
    print(f"\n" + "="*60)
    print("ALL CLUSTERS")
    print("="*60)
    analyzer.print_cluster_summary(analyzer.clusters)
    
    # Save results
    output_file = "/home/ubuntu/mech-interp-color-steering/html_clustering_results.json"
    analyzer.save_results(output_file)
    
    return analyzer, cross_clusters

if __name__ == "__main__":
    analyzer, cross_clusters = main()
