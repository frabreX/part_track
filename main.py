import subprocess


# Avvia il core script come processo indipendente (non si chiude se la GUI crasha)
subprocess.Popen(["python", "core_script.py"], start_new_session=True)

