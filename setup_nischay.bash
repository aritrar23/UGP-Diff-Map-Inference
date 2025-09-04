# 1) Download (x86_64/Linux). Adjust version if needed.
wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz

# 2) Extract and place it somewhere standard
tar -xzf gurobi11.0.3_linux64.tar.gz
sudo mv gurobi1103 /opt/gurobi1103

# 3) Add Gurobi to your PATH and loader paths (shell init)
echo 'export GUROBI_HOME=/opt/gurobi1103/linux64' >> ~/.bashrc
echo 'export PATH=$PATH:$GUROBI_HOME/bin'         >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GUROBI_HOME/lib' >> ~/.bashrc
source ~/.bashrc

# 4) Retrieve and install the license with grbgetkey
grbgetkey <KEY>   # this writes ~/gurobi.lic by default

# 5) (Optional) point explicitly to the license file
echo 'export GRB_LICENSE_FILE=$HOME/gurobi.lic' >> ~/.bashrc
source ~/.bashrc

# 6) Verify
gurobi_cl --version
gurobi_cl --license

