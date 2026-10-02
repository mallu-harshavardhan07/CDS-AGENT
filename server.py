"""
=============================================================================
Module: server.py
Description: Streamlit entrypoint and bridge for SAP CDS Agent Dashboard.
             Enables seamless execution via both:
                 streamlit run server.py
                 streamlit run app.py
                 python server.py
=============================================================================
"""

import os
import sys
import runpy

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

# Detect whether we are inside Streamlit runtime
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    is_streamlit_running = get_script_run_ctx() is not None
except Exception:
    is_streamlit_running = False

if is_streamlit_running or "streamlit" in os.path.basename(sys.argv[0]).lower():
    # Running inside Streamlit: execute app.py
    runpy.run_path(app_path, run_name="__main__")
else:
    # Running via normal Python: launch streamlit run app.py
    import subprocess
    venv_streamlit = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "streamlit.exe")
    cmd = [venv_streamlit if os.path.exists(venv_streamlit) else "streamlit", "run", app_path] + sys.argv[1:]
    subprocess.run(cmd)
