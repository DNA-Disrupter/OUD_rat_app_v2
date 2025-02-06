import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gdown
import os

# ---------------------------
# 1. Load the Data (with caching and download check)
# ---------------------------
@st.cache_data
def load_data():
    output = "normalized_counts_long.csv"
    # Download only if the file doesn't exist
    if not os.path.exists(output):
        file_id = "1sRonWw-MuWNqHctz_VZJTWBOQfujS44f"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

df = load_data()

# ---------------------------
# Helper Function: Darken a Hex Color
# ---------------------------
def darken_color(hex_color, factor=0.7):
    """
    Darkens the given hex color by the given factor (0 < factor < 1).
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, int(r * factor))
    g = max(0, int(g * factor))
    b = max(0, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"

# ---------------------------
# Define Treatment Group Color Mapping
# ---------------------------
color_map = {
    "SALINE": "#1f77b4",        # Blue
    "F-O": "#ff7f0e",           # Orange
    "0 MG/KG": "#9467bd",       # Purple
    "25 MG/KG": "#d62728",      # Red
    "40 MG/KG": "#e377c2",      # Pink
    "70 MG/KG": "#8c564b",      # Brown
    "40 MG/KG w/o SULB": "#2ca02c"  # Green
}

# ---------------------------
# App Title and Description
# ---------------------------
st.title("Bulk RNA-seq OUD Data Visualizer: Median-Ratio Normalization with DESeq2 Approach")

st.markdown(
    """
    **Overview**  

    This app visualizes gene expression data normalized using 
    median‐ratio normalization (the **[DESeq2 method](https://bioconductor.org/packages/release/bioc/html/DESeq2.html)**) 
    for bulk RNA-seq from our OUD study. Unlike TPM normalization—which estimates the relative 
    abundance of transcripts—median‐ratio normalization scales each sample’s read counts so that 
    differences between samples more accurately reflect differences in absolute expression.  

    This method assumes that most genes are expressed at similar levels across samples, so any 
    systematic differences are largely due to library size effects. 
    (**[This approach makes sense biologically](https://mbernste.github.io/posts/median_ratio_norm/)** 
    because most genes serve fundamental cellular functions that remain relatively stable across 
    conditions, meaning that only a subset of genes undergo significant expression changes in response 
    to experimental perturbations). 
    - **Nucleus Accumbens (NAc)**
    - **Prefrontal Cortex (PFC)**
    - **Ventral Tegmental Area (VTA)**
    
    **Treatment Conditions**

    - Saline control  
    - Fentanyl control (F-O)  
    - 0 mg/kg GATC-1021 with 50 mg/kg Sulbutiamine  
    - 25 mg/kg GATC-1021 with 50 mg/kg Sulbutiamine  
    - 40 mg/kg GATC-1021 with 50 mg/kg Sulbutiamine  
    - 70 mg/kg GATC-1021 with 50 mg/kg Sulbutiamine  
    - 40 mg/kg GATC-1021 without Sulbutiamine  
    
    **Sample Composition**

    Except for the 0 MG/KG treatment group, which had 6 female and 8 male samples in the VTA, all other treatment groups had 6 female and 7 male samples 
    across the NAc, PFC, and VTA.

    Use the controls below to select a gene, treatment groups, and brain regions.
    """
)

# ---------------------------
# Sidebar: Brain Region Settings
# ---------------------------
st.sidebar.header("Brain Region Settings")
brain_region_options = ["NAc", "PFC", "VTA"]
selected_brain_regions = st.sidebar.multiselect("Select brain regions:", options=brain_region_options, default=brain_region_options)
view_mode = st.sidebar.radio("View brain regions:", options=["Combined", "Separated"])

# ---------------------------
# 2. Select Gene from a Searchable Dropdown
# ---------------------------
gene_options = sorted(df["Gene"].unique())
selected_gene = st.selectbox("Select a gene:", options=gene_options, 
                             index=gene_options.index("Grin1") if "Grin1" in gene_options else 0)

# ---------------------------
# 3. Select Treatment Groups from a Dropdown
# ---------------------------
treatment_options = sorted(df["GATC.D3"].dropna().unique())
selected_treatments = st.multiselect("Select treatment groups:", options=treatment_options, default=["SALINE", "F-O", "40 MG/KG w/o SULB"])

# ---------------------------
# 4. Filter Data for the Selected Gene, Treatments, and Brain Regions
# ---------------------------
gene_df = df[df["Gene"] == selected_gene].copy()
if selected_treatments:
    gene_df = gene_df[gene_df["GATC.D3"].isin(selected_treatments)].copy()
if selected_brain_regions:
    gene_df = gene_df[gene_df["Profile"].isin(selected_brain_regions)].copy()

if gene_df.empty:
    st.error(f"No data found for gene '{selected_gene}' with the chosen treatment groups and brain regions.")
    st.stop()

# ---------------------------
# 5. Set Up Numeric Mapping for Treatment Groups (and Brain Regions if Separated)
# ---------------------------
if selected_treatments:
    treatment_order = [t for t in selected_treatments if t in gene_df["GATC.D3"].unique()]
else:
    treatment_order = sorted(gene_df["GATC.D3"].unique())
treatment_to_num = {t: i for i, t in enumerate(treatment_order)}

if view_mode == "Separated":
    region_order = [r for r in selected_brain_regions if r in gene_df["Profile"].unique()]
    region_to_num = {r: i for i, r in enumerate(region_order)}
    num_regions = len(region_order)
    offsets = {r: (region_to_num[r] - (num_regions - 1) / 2) * 0.3 for r in region_order}

# ---------------------------
# 6. Create Traces (Box + Scatter Overlay)
# ---------------------------
box_traces = []
scatter_traces = []

if view_mode == "Combined":
    for t in treatment_order:
        df_temp = gene_df[gene_df["GATC.D3"] == t]
        x_vals = [treatment_to_num[t]] * len(df_temp)
        base_color = color_map.get(t, "#7f7f7f")
        dark_color = darken_color(base_color, factor=0.7)
        
        # Create the box trace with low opacity and no built-in points.
        box_traces.append(go.Box(
            x=x_vals,
            y=df_temp["NormalizedCount"],
            name=t,
            marker=dict(color=base_color, opacity=0.3),
            boxpoints=False,
            hoverinfo="skip"
        ))
        # Create the scatter trace for individual points with jitter.
        jitter = np.random.uniform(-0.15, 0.15, size=len(df_temp))
        x_scatter = df_temp["GATC.D3"].map(treatment_to_num) + jitter
        scatter_traces.append(go.Scatter(
            x=x_scatter,
            y=df_temp["NormalizedCount"],
            mode="markers",
            marker=dict(color=dark_color, opacity=1, size=8),
            showlegend=False,
            customdata=df_temp[["Subject.ID", "Sex", "Profile"]].values,
            hovertemplate=(
                "ID: %{customdata[0]}<br>" +
                "Sex: %{customdata[1]}<br>" +
                "Brain Region: %{customdata[2]}<br>" +
                "Treatment: %{text}<br>" +
                "Count: %{y:.2f}<extra></extra>"
            ),
            text=df_temp["GATC.D3"]
        ))
else:
    for t in treatment_order:
        for r in region_order:
            df_temp = gene_df[(gene_df["GATC.D3"] == t) & (gene_df["Profile"] == r)]
            if df_temp.empty:
                continue
            base = treatment_to_num[t]
            offset = offsets[r]
            x_vals = [base + offset] * len(df_temp)
            trace_name = f"{t} - {r}"
            base_color = color_map.get(t, "#7f7f7f")
            dark_color = darken_color(base_color, factor=0.7)
            
            box_traces.append(go.Box(
                x=x_vals,
                y=df_temp["NormalizedCount"],
                name=trace_name,
                marker=dict(color=base_color, opacity=0.3),
                boxpoints=False,
                hoverinfo="skip"
            ))
            jitter = np.random.uniform(-0.15, 0.15, size=len(df_temp))
            x_scatter = df_temp["GATC.D3"].map(treatment_to_num) + offset + jitter
            scatter_traces.append(go.Scatter(
                x=x_scatter,
                y=df_temp["NormalizedCount"],
                mode="markers",
                marker=dict(color=dark_color, opacity=1, size=8),
                showlegend=False,
                customdata=df_temp[["Subject.ID", "Sex", "Profile"]].values,
                hovertemplate=(
                    "ID: %{customdata[0]}<br>" +
                    "Sex: %{customdata[1]}<br>" +
                    "Brain Region: %{customdata[2]}<br>" +
                    "Treatment: %{text}<br>" +
                    "Count: %{y:.2f}<extra></extra>"
                ),
                text=df_temp["GATC.D3"]
            ))

# ---------------------------
# 7. Combine Traces and Update the Layout
# ---------------------------
fig = go.Figure(data=box_traces + scatter_traces)
if view_mode == "Combined":
    tickvals = list(treatment_to_num.values())
    ticktext = list(treatment_to_num.keys())
else:
    tickvals = list(treatment_to_num.values())
    ticktext = list(treatment_to_num.keys())

fig.update_layout(
    title=f"Expression of {selected_gene} by Treatment Group",
    xaxis=dict(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        title="Treatment Group"
    ),
    yaxis=dict(title="Normalized Count (Median-of-Ratios)"),
    template="simple_white"
)

st.plotly_chart(fig, use_container_width=True)
