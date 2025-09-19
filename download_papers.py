#!/usr/bin/env python3
"""
Download essential papers from ArXiv for Early Stopping research
"""

import os
import time
import requests
from typing import List, Tuple

def download_paper(arxiv_id: str, filename: str, output_dir: str = "Articles") -> bool:
    """Download a single paper from ArXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"[OK] {filename} already exists")
        return True
    
    try:
        print(f"[DOWNLOADING] {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[SUCCESS] Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return False

def main():
    """Download all essential papers"""
    
    # Create output directory
    os.makedirs("Articles", exist_ok=True)
    
    # List of papers to download (arxiv_id, filename)
    papers: List[Tuple[str, str]] = [
        # Core References
        ("2008.02752", "01_Celisse_Wahl_2021_Discrepancy_Principle.pdf"),
        ("2406.15001", "02_Hucker_Reiss_2024_Conjugate_Gradients.pdf"),
        ("1606.07702", "03_Blanchard_2018_Optimal_Adaptation.pdf"),
        ("1306.3574", "04_Raskutti_2014_Optimal_Stopping_Rule.pdf"),
        
        # RKHS and Regularization
        ("1405.0042", "05_Rosasco_Villa_2015_Incremental_Regularization.pdf"),
        
        # Gradient Descent
        ("1810.10082", "06_Ali_2019_Continuous_Time_Early_Stopping.pdf"),
        ("1805.11921", "07_Suggala_2018_Optimization_Regularization_Paths.pdf"),
        
        # Boosting
        ("math/0508276", "08_Zhang_Yu_2005_Boosting_Early_Stopping.pdf"),
        ("1603.02754", "09_Chen_Guestrin_2016_XGBoost.pdf"),
        
        # Deep Learning
        ("1705.09280", "10_Gunasekar_2018_Implicit_Regularization.pdf"),
        ("1706.08947", "11_Neyshabur_2017_Generalization_Deep_Learning.pdf"),
        
        # Privacy
        ("1405.7085", "12_Bassily_2014_Private_ERM.pdf"),
        ("1607.00133", "13_Abadi_2016_Deep_Learning_Differential_Privacy.pdf"),
        
        # Fairness
        ("1801.04849", "14_Cotter_2019_Fairness_Constraints.pdf"),
        ("1803.02453", "15_Agarwal_2018_Fair_Classification.pdf"),
        
        # Federated Learning
        ("1602.05629", "16_McMahan_2017_Federated_Learning.pdf"),
        ("1908.07873", "17_Li_2020_Federated_Learning_Survey.pdf"),
        
        # Frugal AI
        ("1907.10597", "18_Schwartz_2020_Green_AI.pdf"),
        ("2106.08962", "19_Menghani_2023_Efficient_Deep_Learning.pdf"),
        
        # OpenML
        ("1407.7722", "20_Vanschoren_2014_OpenML.pdf"),
    ]
    
    print("=" * 60)
    print("DOWNLOADING ESSENTIAL EARLY STOPPING PAPERS FROM ARXIV")
    print("=" * 60)
    print(f"Total papers to download: {len(papers)}\n")
    
    success_count = 0
    failed_papers = []
    
    for i, (arxiv_id, filename) in enumerate(papers, 1):
        print(f"\n[{i}/{len(papers)}] Paper: {filename}")
        
        if download_paper(arxiv_id, filename):
            success_count += 1
        else:
            failed_papers.append(filename)
        
        # Be polite to ArXiv servers
        if i < len(papers):
            time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"[COMPLETE] Successfully downloaded: {success_count}/{len(papers)}")
    
    if failed_papers:
        print(f"[FAILED] Failed downloads ({len(failed_papers)}):")
        for paper in failed_papers:
            print(f"  - {paper}")
        print("\nYou can manually download these from:")
        print("  https://arxiv.org/")
    else:
        print("All papers downloaded successfully!")
    
    print("\n[INFO] Papers saved in: Articles/")

if __name__ == "__main__":
    main()