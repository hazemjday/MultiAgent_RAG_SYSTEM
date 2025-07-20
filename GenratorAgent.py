import torch
import faiss
import pickle
import re
import numpy as np
from typing import Dict
from fpdf import FPDF
from transformers import pipeline

class Generator:
    def __init__(self, min_length=50):
        self.min_length = min_length
        self.exclude_titles = [
            "External links", "References", "Further reading", 
            "See also", "Footnotes", "Bibliography"
        ]
        self.section_pattern = re.compile(r'^[A-Z][A-Za-z0-9\s]+[^\.,;:?!]$')

    def split_into_sections(self, text: str) -> Dict[str, str]:
        lines = text.split('\n')
        sections = {}
        current_section = "Introduction"
        sections[current_section] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self.section_pattern.match(line):
                current_section = line
                sections[current_section] = []
            else:
                sections[current_section].append(line)
        for sec in list(sections.keys()):
            sections[sec] = " ".join(sections[sec])
            if not sections[sec].strip() or sec in self.exclude_titles or len(sections[sec]) < self.min_length:
                del sections[sec]
        return sections

    def resume(self, text: str, title: str) -> str:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
        safe_query = re.sub(r'\W+', '_', title.lower())
        sections = self.split_into_sections(text)
        if not sections:
            return "No valid sections found."
        
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        # Titre principal
        pdf.set_font("DejaVu", size=16)
        pdf.set_text_color(0, 0, 255)
        pdf.cell(0, 10, title, ln=1, align='C')
        pdf.ln(10)

        # Corps du PDF
        pdf.set_font("DejaVu", size=12)
        for key, value in sections.items():
            pdf.set_text_color(0, 0, 255)  # 🔵 Bleu pour la clé
            pdf.cell(0, 10, f"{key}:", ln=1)
            pdf.set_text_color(0, 0, 0)    # ⚫ Noir pour la valeur
            resume_value = summarizer(value, min_length=20, do_sample=False)
            pdf.multi_cell(0, 10, resume_value[0]["summary_text"])
            pdf.ln(5)
        filename = f"rapport_{safe_query}.pdf"
        # Sauvegarde du PDF
        pdf.output(filename)
        return f"Résumé sauvegardé dans {filename}"






