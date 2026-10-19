import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, 
    r2_score
)

def convert_from_z_score(z_vals, sd, mean):
    """
    Converts z-score back to original value using mean and sd

    X = Z * standard_deviation + mean

    :param z_vals: z-score values to be converted
    :param sd: standard deviation of original data
    :param mean: mean of original data
    :return: input values converted to back to original units
    """

    z_score_val = z_vals * sd + mean

    return z_score_val

def read_reg_obs_pred(labels_fpath, obs_pred_fpath):

    # Read predictions and make plot ID uppercase
    df = (pd.read_csv(obs_pred_fpath)
                .assign(plot_id=lambda df: df['plot_id'].str.upper()))

    # Read labels
    labels_df = (pd.read_csv(labels_fpath)
                .assign(plot_id=lambda df: df['plot_id'].str.upper())
                .rename(columns={'total_agb_mg_ha': 'agb_mg_ha_obs'}))

    # Drop Z-scored AGB labels from dataframes
    df.pop('total_agb_z')
    labels_df.pop('total_agb_z')

    # Join with labels DF
    df = df.merge(labels_df, on='plot_id', how='left')

    # Set mean and standard deviation for Z-score conversion
    mean_agb = df['agb_mg_ha_obs'].mean()
    sd_agb = df['agb_mg_ha_obs'].std()

    # Convert predicted Z-scores back to original AGB values (tonnes per hectare)
    df['agb_mg_ha_pred'] = convert_from_z_score(z_vals=df['total_agb_z_pred'],
                                            sd=sd_agb,
                                            mean=mean_agb)
    
    return df

def plot_regression(obs, pred, ax_lims, title='', fig_out_fpath=None, hide=False):

    # Calculate regression metrics
    r2 = r2_score(obs, pred)
    mse = mean_squared_error(obs, pred)
    rmse = np.sqrt(mse)
    bias = np.mean(pred - obs)
    n_samples = len(obs)

    # Set square figure
    fig, ax = plt.subplots(figsize=(5, 5))  # Make figure square
    ax.set(aspect='equal')

    # Set axis limits to be equal and tight
    ax.set_xlim(ax_lims[0], ax_lims[1])
    ax.set_ylim(ax_lims[0], ax_lims[1])
 
    sns.scatterplot(ax=ax, x=obs, y=pred, color='black')
    ax.plot([ax_lims[0], ax_lims[1]], [ax_lims[0], ax_lims[1]], color='red', linestyle='--')
    ax.set_xlabel('Observed AGB (Mg/ha)', fontsize=11)
    ax.set_ylabel('Predicted AGB (Mg/ha)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.text(.02, .98, 
            s = f'R²: {r2:.2f}\nRMSE: {rmse:.2f} Mg/ha\nBias: {bias:.2f} Mg/ha\nn: {n_samples}',
            ha='left', va='top', 
            transform=ax.transAxes,
            fontsize=11)

    fig.tight_layout()

    if fig_out_fpath:
        fig.savefig(fig_out_fpath, dpi=400, bbox_inches='tight', pad_inches=0.05)
    
    if hide:
        plt.close(fig)
    else:
        plt.show()