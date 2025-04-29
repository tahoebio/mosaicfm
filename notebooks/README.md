Notebooks stored in directories split according to dataset.
Recent experiments under the Tahoe directory:
- `vevo_tahoe_wilcoxon_filter_count.ipynb`: filter by genes expressed by n number of cells; use embedding-based method.
- `vevo_tahoe_wilcoxon_filter_protein_coding.ipynb`: filter by protein-coding genes; use embedding-based method.
- `vevo_tahoe_wilcoxon_hvg.ipynb`: filter by HVG; use embedding based method.
- `vevo_tahoe_wilcoxon_filter_hvg_attn.ipynb`: filter by HVG; use attention-based method.

`vevo_tahoe_wilcoxon_filter_count.ipynb` and `vevo_tahoe_wilcoxon_filter_hvg_attn.ipynb` calculates results cumulatively by randomly partitioning the adata and iterates through every partition. A png containing a box-and-whisker plot will be saved for every iteration once that iteration contains control profiles. The cumulative partition size and the scGPT rank mean for that iteration will be included in the filename of the png.