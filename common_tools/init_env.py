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
  !pip install netCDF4 gdown palettable >> log.txt
  !pip install -q condacolab >> log.txt
  import condacolab
  condacolab.install()
  !conda install -c conda-forge esmpy >> log.txt
  print('....loading....')
  !conda install -c conda-forge xesmf >> log.txt
  !git clone https://github.com/CNES/lenapy.git >> log.txt
  !pip install lenapy/. >> log.txt
  print('.... Done')


def load_data_WP85():
  print("Load GMSL timeserie\n")
  !gdown --id {"1sWWo6zlh3qYB13zOKneiklBWhsgT8XWW"} -O MSL_aviso.nc >> log.txt
  gmsl_file = 'MSL_aviso.nc'

  print("Load TOPEX-A correction\n")
  !gdown --id {"1e_r15fM16UwzmkqcxS4OUhDvrQ3U21fl"} -O j3_wtc_drift_correction_cdr_al_s3a.nc >> log.txt
  tpa_corr_file = 'MSL_Aviso_Correction_GMSL_TPA.nc'

  print("Load Jason-3 correction\n")
  !gdown --id {"1HQq52w2NrM8Xsm0Nsye4Q7BEHAJhUa07"} -O j3_wtc_drift_correction_cdr_al_s3a.nc >> log.txt
  j3_corr_file = 'j3_wtc_drift_correction_cdr_al_s3a.nc'

  print("Load table of budget errors\n")
  r = requests.get('https://drive.google.com/uc?export=download&id=110SsJUTu3wBKhc6OHuun5bNDz08tm3eJ')
  with open('error_budget_table.yaml', 'wb') as f:
      f.write(r.content)
  error_prescription = 'error_budget_table.yaml'
  
  return gmsl_file, tpa_corr_file, j3_corr_file, error_prescription
