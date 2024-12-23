# %% load results from file
df_results = pd.read_csv('../../data/processed/ga_hpo_sent_embedding_results.csv')

# %% plot results
# using the results dictionary and seaborne create a violon plot with a plot for: 1. gene to solution similarity and  2. patient to solution similarity
pp = PdfPages('./output/figures/shared-phenotype-similarity-distributions.pdf')
sns.set_theme(style="darkgrid")

plt.figure(figsize=(10,6))
# set the violon plot colors to be dodger blue and orange
sns.violinplot(data=df_results[['gene_solution_similarity', 'patient_solution_mean_similarity', 'patient_gene_mean_similarity']],
               palette="Blues")
# add a dark grey dashed line for the mean of the gene to random similarity 
plt.axhline(np.mean(df_results['gene_mean_random_similarity']), color='k', linestyle='--')
# add a line for the mean of the patient to random similarity
plt.axhline(np.mean(df_results['patient_mean_random_similarity']), color='k')
# change the x-axis labels to be more descriptive
plt.xticks([0,1,2], ['Gene \u2194 Solution', 'Population \u2194 Solution', 'Gene \u2194 Population'])
# make the x tick lables larger
plt.xticks(fontsize=14)
# add a title
plt.title('Phenotype Similarity Distributions', fontsize=16)
# make the y-ticks larger
plt.yticks(fontsize=14)
# save the figure as a pdf
pp.savefig()
pp.close()

# %%
# create a scatter plot of the mean patient term count vs the solution term count
pp = PdfPages('./output/figures/patient-term-count-vs-solution-term-count.pdf')
plt.scatter(df_results['patient_mean_term_count'], df_results['solution_term_count'])
plt.xlabel('Mean Per Patient (grouped by gene)', fontsize=14)
plt.ylabel('Shared Phenotype', fontsize=14)
plt.title('HPO Term Count by Gene Population', fontsize=16)
# make the x and y ticks have the same range of values
# set the x and y limits to be the same
plt.xlim([0, 70])
plt.ylim([0, 70])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid()
pp.savefig()
pp.close()

# %%
# plot the solution similarity as a function of the number of terms in the gene
pp = PdfPages('./output/figures/gene-term-count-vs-solution-similarity.pdf')
plt.scatter(df_results['gene_term_count'], df_results['gene_solution_similarity'])
# add a trend line
z = np.polyfit(df_results['gene_term_count'], df_results['gene_solution_similarity'], 1)
p = np.poly1d(z)
plt.plot(df_results['gene_term_count'],p(df_results['gene_term_count']),"r--")
# add text with the correlation coefficient
plt.text(-0.5, 0.9, f'Correlation Coefficient: {np.corrcoef(df_results["gene_term_count"], df_results["gene_solution_similarity"])[0,1]:.2f}', fontsize=14)

plt.xlabel('Gene Term Count', fontsize=14)
plt.ylabel('Shared Phenotype', fontsize=14)
plt.title('Gene Term Count vs Solution Similarity', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
pp.savefig()
pp.close()

# %%
df_results[['gene_id', 'gene_term_count', 'gene_terms', 'solution_terms']].sort_values(by='gene_term_count').head()
# %%
