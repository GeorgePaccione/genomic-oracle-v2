import spaces  # MUST BE LINE 1. Fixes the "CUDA Initialized" error!
import gradio as gr
import os
import re
import requests
import torch
import joblib
import numpy as np
import torch.nn.functional as F
from Bio.Blast import NCBIWWW, NCBIXML

# include HF native imports for the Phenotype model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BertTokenizer, BertForSequenceClassification, AutoConfig
from huggingface_hub import hf_hub_download

# ===================================
# 1. LOAD AI MODELS (GLOBALLY CACHED)
# ===================================
print("Waking up the Genomic Oracle...\n")

# A. Kadir's Gatekeeper 
clf_coding = joblib.load("coding_classifier_universal.joblib")

# B. Base DNABERT 
tokenizer_base = AutoTokenizer.from_pretrained("DNABERT_Local", trust_remote_code=True)
model_base = AutoModel.from_pretrained("DNABERT_Local", trust_remote_code=True, _fast_init=False)
model_base.eval()

# C. DNABERT-2 Promoter Model 
tokenizer_promoter = AutoTokenizer.from_pretrained("llm_promoter_classifier_v2", trust_remote_code=True)
model_promoter = AutoModelForSequenceClassification.from_pretrained("llm_promoter_classifier_v2", trust_remote_code=True, _fast_init=False)
model_promoter.eval()

# D. Multi-Feature LightGBM 
lgbm_path = hf_hub_download(repo_id="Geonomic/Genomic-Oracle-Weights", filename="dnabert_lightgbm_model_feature_type_v2.pkl")
raw_lgbm = joblib.load(lgbm_path)

# If it's a dictionary, print the keys to the log and try to extract the model
if isinstance(raw_lgbm, dict):
    print(f" DEBUG: LightGBM Dictionary Keys: {raw_lgbm.keys()}")
    # We will try the most common names for saved models
    if "model" in raw_lgbm:
        lightgbm_model = raw_lgbm["model"]
    elif "classifier" in raw_lgbm:
        lightgbm_model = raw_lgbm["classifier"]
    else:
        # Fallback: just grab the very first thing in the dictionary
        first_key = list(raw_lgbm.keys())[0]
        lightgbm_model = raw_lgbm[first_key]
else:
    lightgbm_model = raw_lgbm

# E. Custom Lean/Obese Phenotype BERT (Forced Native Architecture via Colab Fix)
tokenizer_pheno = BertTokenizer.from_pretrained("Geonomic/Genomic-Oracle-Weights", do_lower_case=False)
config_pheno = AutoConfig.from_pretrained("Geonomic/Genomic-Oracle-Weights", trust_remote_code=True)
model_pheno = BertForSequenceClassification.from_pretrained("Geonomic/Genomic-Oracle-Weights", config=config_pheno, _fast_init=False)
model_pheno.eval()

FEATURE_DICT = {
    0: "Gene/Transcript (Coding/mRNA)",
    1: "Regulatory Region (Promoter/Enhancer/Silencer)",
    2: "Long Non-Coding RNA (lncRNA)",
    3: "Small/Transfer RNA (snRNA/miRNA/tRNA)",
    4: "Repeat Region / Mobile Genetic Element",
    5: "Pseudogene"
}

# ==============================================
# 2. CORE INFERENCE ENGINE (ZeroGPU Accelerated)
# ==============================================
@spaces.GPU  
def run_deep_learning_cascade(dna_sequence):
    device = torch.device("cuda")

    # THE FINAL KEY: Teleport the CPU-locked models into the A100 GPU!
    model_base.to(device)
    model_promoter.to(device)
    model_pheno.to(device)
    clean_seq = "".join(dna_sequence.split()).upper()
    
    # --- LEVEL 1: Base Embedding & Kadir's Gatekeeper ---
    inputs = tokenizer_base([clean_seq], return_tensors="pt", max_length=300, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        out_base = model_base(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (out_base[0] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vector = embedding.float().cpu().numpy()

    p_coding = clf_coding.predict_proba(vector)[0][1]
    is_coding = p_coding >= 0.60
    
    raw_scores = {"Protein-Coding Probability": p_coding}

    # --- LEVEL 2: LightGBM Structural Classification ---
    lgb_prediction = int(lightgbm_model.predict(vector)[0])
    structural_feature = FEATURE_DICT.get(lgb_prediction, "Unknown Region")
    # raw_scores["Predicted Structure"] = structural_feature

    # THE CONTRADICTION RESOLVER
    # The fine-tuned LightGBM overrides any contradiction during class assignment
    if lgb_prediction == 0:
        is_coding = True
        # If LightGBM had to overrule, we boost the base confidence to match its high AUROC accuracy
        confidence = p_coding if p_coding >= 0.50 else 0.85 
    else:
        is_coding = False
        confidence = (1 - p_coding) if p_coding < 0.50 else 0.85

# --- LEVEL 3: The Deep Learning Branching Logic ---
    summary_dict = {
        "Final Classification": "GENE" if is_coding else "NON-CODING",
        "Feature": structural_feature
    }

    # BRANCH A: Phenotype Analysis (Triggered if Coding AND is CDS/Exon)
    if is_coding and lgb_prediction == 0:
        kmers = [clean_seq[i:i+5] for i in range(len(clean_seq) - 4)]
        spaced_kmers = " ".join(kmers)
        inputs_pheno = tokenizer_pheno(spaced_kmers, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model_pheno(**inputs_pheno)
            probs = F.softmax(outputs.logits, dim=-1)
            prob_obese, prob_lean = probs[0][0].item(), probs[0][1].item()
            
            phenotype = "Obesity-Associated" if prob_obese > prob_lean else "Lean-Associated"
            summary_dict["Phenotype"] = phenotype
            raw_scores["Phenotype (Obese)"] = prob_obese
            raw_scores["Phenotype (Lean)"] = prob_lean

    # BRANCH B: Promoter Validation (Triggered if Non-Coding AND is Promoter/Enhancer)
    elif not is_coding and lgb_prediction == 1:
        inputs_promo = tokenizer_promoter([clean_seq], return_tensors="pt", max_length=300, truncation=True, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model_promoter(**inputs_promo)
            probs = F.softmax(outputs.logits, dim=-1)
            p_promoter = probs[0][0].item()
            
            validation = "High Confidence Regulatory Element" if p_promoter >= 0.50 else "Weak Regulatory Signal"
            summary_dict["Validation"] = validation
            raw_scores["Promoter Signal"] = p_promoter

    return summary_dict, confidence, raw_scores

# ===================================
# 3. SPATIAL MAPPING (NCBI / ENSEMBL)
# ===================================
def get_genomic_context(sequence, is_coding):
    feature_type = "CODING" if is_coding else "PROMOTER"
    try:
        # ask blast for 5 hits instead of 1 so we can hunt for the true chromosome
        result_handle = NCBIWWW.qblast(
            "blastn", 
            "nt", 
            sequence, 
            entrez_query="Homo sapiens[Organism] AND biomol_genomic[PROP]", 
            hitlist_size=50
        )
        blast_record = NCBIXML.read(result_handle)
    except Exception as e:
        return {"error": f"BLAST Connection Error: {e}"}

    if not blast_record.alignments:
        return {"error": "No human genome match found for this sequence."}

    # Loop through the top hits and grab the first one that is an actual Chromosome
    alignment = blast_record.alignments[0] # Default to the top hit
    chrom = None
    for aln in blast_record.alignments:
        chrom_match = re.search(r"chromosome\s([0-9XYMT]+)", aln.title, re.IGNORECASE)
        if chrom_match:
            alignment = aln
            chrom = chrom_match.group(1)
            break # We found the chromosome, stop searching!

    hsp = alignment.hsps[0]
    location_string = f"Chromosome {chrom}" if chrom else f"Accession {alignment.accession}"

    start, end = hsp.sbjct_start, hsp.sbjct_end
    is_forward = (start < end)
    strand_txt = "Forward (+)" if is_forward else "Reverse (-)"

    if chrom is None:
        return {"location": location_string, "start": start, "end": end, "strand": strand_txt, "metadata": "BLAST returned a localized record without a chromosome. Ensembl mapping skipped."}

    search_start = min(start, end) if feature_type == "CODING" else (end if is_forward else max(1, end - 15000))
    search_end = max(start, end) if feature_type == "CODING" else (end + 15000 if is_forward else end)

    try:
        response = requests.get(
            f"https://rest.ensembl.org/overlap/region/human/{chrom}:{search_start}-{search_end}?feature=gene", 
            headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        genes = response.json()
    except Exception as e:
        return {"location": location_string, "start": start, "end": end, "strand": strand_txt, "metadata": f"Ensembl mapping unavailable: {e}"}

    if not genes:
        gene_desc = "No annotated genes found in this specific region."
    else:
        if feature_type == "PROMOTER":
            genes.sort(key=lambda x: min(abs(x['start'] - end), abs(x['end'] - end)))
        top_gene = genes[0]
        name = top_gene.get('external_name', 'Unknown')
        biotype = top_gene.get('biotype', 'Unknown').replace('_', ' ').title()
        desc = top_gene.get('description', 'No description available.').split(' [')[0]
        gene_desc = f"Matches Gene: {name} | Type: {biotype} | Function: {desc}" if feature_type == "CODING" else f"Regulates Downstream Gene: {name} | Type: {biotype} | Function: {desc}"

    return {"location": location_string, "start": start, "end": end, "strand": strand_txt, "metadata": gene_desc}

# ==============================
# 4. GRADIO INTERFACE (FRONTEND)
# ==============================
def gradio_inference(dna_sequence, run_mapping):
    if len(dna_sequence.strip()) < 10:
        error_html = f"""
        <div style="border: 1px solid #dc3545; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
            <h3 style="margin-top: 0; margin-bottom: 10px; color: #dc3545;">❌ Classification Summary</h3>
            <div style="font-size: 1.15em;">Sequence too short! Please enter at least 10 base pairs.</div>
        </div>
        """
        # Updated the placeholder text here as well!
        yield (error_html, "<div style='border: 1px solid #4b5563; border-radius: 8px; padding: 15px; margin-bottom: 15px;'><h3 style='margin-top: 0; margin-bottom: 10px;'>📊 Internal Pipeline Statistics</h3><div style='color: #9ca3af; font-style: italic;'>Results will appear here...</div></div>", "", "")
        return

    # Run AI Models (GPU)
    summary_dict, conf, raw_scores = run_deep_learning_cascade(dna_sequence)

    # Build Custom HTML for Stats 
    stats_html = """
    <div style="border: 1px solid #4b5563; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
        <h3 style="margin-top: 0; margin-bottom: 10px;"> Internal Pipeline Statistics</h3>
        <div style="font-size: 1.15em; line-height: 1.8;">
    """
    for key, val in raw_scores.items():
        if isinstance(val, float):
            # Added color: #10b981; to make all percentage scores GREEN
            stats_html += f"{key}: <b style='color: #10b981;'>{val:.2%}</b><br>"
        else:
            stats_html += f"{key}: <b>{val}</b><br>"
    stats_html += "</div></div>"

    # Build Custom HTML for Summary 
    summary_html = f"""
    <div style="border: 1px solid #4b5563; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
        <h3 style="margin-top: 0; margin-bottom: 10px;"> Classification Summary <span style="color: #0d6efd; font-size: 0.85em; font-weight: normal;">(Deep Scan Complete)</span></h3>
        <div style="font-size: 1.15em; line-height: 1.8">
            Final Classification: <b style="color: #10b981;">{summary_dict.get('Final Classification')}</b><br>
            Feature: <b>{summary_dict.get('Feature')}</b><br>
    """
    if "Phenotype" in summary_dict:
        summary_html += f"Phenotype: <b>{summary_dict['Phenotype']}</b><br>"
    if "Validation" in summary_dict:
        summary_html += f"Validation: <b>{summary_dict['Validation']}</b><br>"
    
    # Confidence score remains green
    summary_html += f"Confidence Score: <b style='color: #10b981;'>{conf:.2%}</b></div></div>"

    if not run_mapping:
        yield (summary_html, stats_html, "Spatial mapping skipped (Enable NCBI query to run).", "")
        return

    # Pushes AI results instantly while showing a loading message for BLAST!
    yield (summary_html, stats_html, "⏳ Querying NCBI BLAST... (This takes 1-3 minutes. Please wait.)", "")

    # Run Context Mapping (CPU / Network)
    is_coding = summary_dict.get("Final Classification") == "GENE"
    context = get_genomic_context(dna_sequence, is_coding)
    
    if "error" in context:
        context_output = f"❌ Mapping failed: {context['error']}"
    elif "location" in context:
        context_lines = [
            f"Location:\t{context['location']}",
            f"Strand:\t{context['strand']}",
            f"Coordinates: {context['start']:,} – {context['end']:,}",
            f"Notes:\t{context['metadata']}"
        ]
        context_output = "\n".join(context_lines)
    else:
        context_output = "⚠️ Could not map sequence."

    # Pushes the finished BLAST results!
    yield (summary_html, stats_html, context_output, "")

# --- CUSTOM CSS ---
custom_css = """
#scan_btn {
    background-color: #0d6efd !important; /* Deep Blue */
    color: white !important;
    border: none !important;
    transition: 0.3s ease;
}
#scan_btn:hover {
    background-color: #dc3545 !important; /* Striking Red */
}
"""

# --- THE UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), title="🧬 The Genomic Oracle 🧬", css=custom_css) as demo:
    
    # 1. The Custom HTML Title
    gr.HTML(
        """
        <div style="text-align: center; padding-bottom: 10px;">
            <h1 style="font-size: 3.5rem; font-weight: bold; margin-bottom: 0.2rem;">🧬 The Genomic Oracle 🧬</h1>
            <h3 style="margin-top: 0; font-weight: normal;"><b>University of Maryland Global Campus</b> | Bioinformatics Capstone</h3>
        </div>
        <hr>
        """
    )
    
    # 2. The Standard Markdown Text
    gr.Markdown(
        """
        Welcome to the official interface for **The Genomic Oracle**, a cascaded machine learning pipeline designed for high-precision DNA sequence classification. 
        
        ### The 4-Stage Cascading Architecture
        1. **The Gatekeeper:** Logistic Regression model rapidly screens native k-mer vectors to identify protein-coding vs. non-coding potential.
        2. **Structural Mapper:** LightGBM model classifies the sequence into 1 of 6 structural features (e.g., lncRNAs, Enhancers, mobile elements).
        3. **Phenotype Prediction:** Sequences flagged as Coding are passed through a custom ALiBi BERT transformer to predict specific traits.
        4. **Regulatory Validation:** Sequences flagged as Promoters or Enhancers are routed to a DNABERT-2 spatial attention neural network.
        
        ---
        Created by: Kadir Galindo, Duncan Hall, Rebecca Mellinger & George Paccione\n
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            dna_input = gr.Textbox(label="Enter DNA Sequence", placeholder="e.g., ATGCGATCGATCGATCG...", lines=10)
            run_mapping_cb = gr.Checkbox(value=False, label="Query NCBI BLAST for spatial mapping (Takes 1–3 mins)")
            submit_btn = gr.Button("🚀 Initialize Deep Scan", elem_id="scan_btn")

        with gr.Column(scale=1):
            # Create default placeholder HTML so the boxes are visible on startup!
            default_summary = """
            <div style="border: 1px solid #4b5563; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h3 style="margin-top: 0; margin-bottom: 10px;">Classification Summary</h3>
                <div style="color: #9ca3af; font-style: italic;">Results will appear here...</div>
            </div>
            """
            
            default_stats = """
            <div style="border: 1px solid #4b5563; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h3 style="margin-top: 0; margin-bottom: 10px;">Internal Pipeline Statistics</h3>
                <div style="color: #9ca3af; font-style: italic;">Results will appear here...</div>
            </div>
            """

            # Pass the default HTML into the components
            output_summary = gr.HTML(value=default_summary)
            stats_panel = gr.HTML(value=default_stats)
            
            mapping_section = gr.Accordion("Genomic Context (BLAST/Ensembl)", open=False)
            with mapping_section:
                context_output = gr.Textbox(label="Mapping Results", lines=5, placeholder="Results will appear here...")

    info_box = gr.Markdown("", elem_id="info_box")

    # NEW UX FEATURE: Auto-open the accordion when the user checks the BLAST box!
    run_mapping_cb.change(
        fn=lambda is_checked: gr.Accordion(open=is_checked),
        inputs=[run_mapping_cb],
        outputs=[mapping_section]
    )

    # Pinned directly to the generator function to allow live-streaming!
    submit_btn.click(
        fn=gradio_inference,
        inputs=[dna_input, run_mapping_cb],
        outputs=[output_summary, stats_panel, context_output, info_box]
    )

demo.launch()
