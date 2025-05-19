pip install torch --index-url https://download.pytorch.org/whl/cu118 
pip install paperswithcode-client>=0.3.1 # install it manually first, then install this list 
pip uninstall lm_eval -y # uninstall current installation first
pip install -r requirements.txt
# pip install git+https://${GITHUB_TOKEN}@github.com/allenai/exec_utils.git