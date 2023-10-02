import sys,os
import getpass

from pathlib import Path

if getpass.getuser() == 'samuel' and  sys.platform.startswith('linux'):
    base_folder = '/home/samuel/mnt/CRNLDATA//crnldata/cmo/Projets/Autre/NBuonviso2022_Sleep_EEG_Respi_Valentin/'
    
elif getpass.getuser() == 'samuel.garcia' and  sys.platform.startswith('linux'):
    base_folder = '/crnldata/cmo/Projets/Autre//NBuonviso2022_Sleep_EEG_Respi_Valentin/'

elif getpass.getuser() == 'valentin' and  sys.platform.startswith('linux'):
    base_folder = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Autre//NBuonviso2022_Sleep_EEG_Respi_Valentin/'
    
elif getpass.getuser() == 'valentin.ghibaudo' and  sys.platform.startswith('linux'):
    base_folder = '/crnldata/cmo/Projets/Autre//NBuonviso2022_Sleep_EEG_Respi_Valentin/'

elif sys.platform.startswith('win'):
	base_folder = 'N:/cmo/Projets/Autre//NBuonviso2022_Sleep_EEG_Respi_Valentin/'


base_folder = Path(base_folder)
article_folder = base_folder / 'autres' / 'article_N20' / 'full_paper'
precomputedir = base_folder / 'precompute'
