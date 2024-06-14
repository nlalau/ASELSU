import os
import sys
from subprocess import check_call
import requests
import matplotlib as mpl
import gdown

def config_plot():
    nice_fonts = {
      "font.family": "serif",
      #"font.weight":"bold",
      "figure.titleweight":"bold",
      "figure.titlesize":20,
      "axes.labelsize": 14,
      "font.size": 16,
      "legend.fontsize": 10,
      "xtick.labelsize": 14,
      "ytick.labelsize": 14,
      }
    
    mpl.rcParams.update(nice_fonts)

def run_command(command):
    with open("log.txt", "a") as log_file:
        check_call(command, shell=True, stdout=log_file, stderr=log_file)

def load_environment_WP85():
    print('Environment loading....')
    run_command("pip install netCDF4 gdown palettable")
    run_command("pip install -q condacolab")
    import condacolab
    condacolab.install()
    run_command("conda install -c conda-forge esmpy -y")
    print('....loading....')
    run_command("conda install -c conda-forge xesmf -y")
    
    # Clone the repository and install lenapy
    run_command("git clone https://github.com/CNES/lenapy.git")
    run_command("pip install lenapy/.")
    print('.... Done')

def load_data_WP85():
    print("Load GMSL timeserie\n")
    gmsl_file_id = "1sWWo6zlh3qYB13zOKneiklBWhsgT8XWW"
    gmsl_output = "MSL_aviso.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={gmsl_file_id}", gmsl_output, quiet=False)
    gmsl_file = gmsl_output

    print("Load TOPEX-A correction\n")
    tpa_corr_file_id = "1e_r15fM16UwzmkqcxS4OUhDvrQ3U21fl"
    tpa_corr_output = "MSL_Aviso_Correction_GMSL_TPA.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={tpa_corr_file_id}", tpa_corr_output, quiet=False)
    tpa_corr_file = tpa_corr_output

    print("Load Jason-3 correction\n")
    j3_corr_file_id = "1HQq52w2NrM8Xsm0Nsye4Q7BEHAJhUa07"
    j3_corr_output = "j3_wtc_drift_correction_cdr_al_s3a.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={j3_corr_file_id}", j3_corr_output, quiet=False)
    j3_corr_file = j3_corr_output

    print("Load table of budget errors\n")
    error_budget_url = 'https://drive.google.com/uc?export=download&id=110SsJUTu3wBKhc6OHuun5bNDz08tm3eJ'
    error_prescription = 'error_budget_table.yaml'
    r = requests.get(error_budget_url)
    with open(error_prescription, 'wb') as f:
        f.write(r.content)
    
    return gmsl_file, tpa_corr_file, j3_corr_file, error_prescription
