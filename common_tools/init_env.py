import os
import sys
from subprocess import check_call

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
  run_command("gdown --id {"1sWWo6zlh3qYB13zOKneiklBWhsgT8XWW"} -O MSL_aviso.nc")
  gmsl_file = 'MSL_aviso.nc'

  print("Load TOPEX-A correction\n")
  run_command("gdown --id {"1e_r15fM16UwzmkqcxS4OUhDvrQ3U21fl"} -O j3_wtc_drift_correction_cdr_al_s3a.nc")
  tpa_corr_file = 'MSL_Aviso_Correction_GMSL_TPA.nc'

  print("Load Jason-3 correction\n")
  run_command("gdown --id {"1HQq52w2NrM8Xsm0Nsye4Q7BEHAJhUa07"} -O j3_wtc_drift_correction_cdr_al_s3a.nc")
  j3_corr_file = 'j3_wtc_drift_correction_cdr_al_s3a.nc'

  print("Load table of budget errors\n")
  r = requests.get('https://drive.google.com/uc?export=download&id=110SsJUTu3wBKhc6OHuun5bNDz08tm3eJ')
  with open('error_budget_table.yaml', 'wb') as f:
      f.write(r.content)
  error_prescription = 'error_budget_table.yaml'
  
  return gmsl_file, tpa_corr_file, j3_corr_file, error_prescription
