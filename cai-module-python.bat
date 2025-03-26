pip uninstall -y numpy pandas scikit-learn xgboost lightgbm tensorflow scipy
pip cache purge

pip install --no-cache-dir numpy==1.23.5
pip install --no-cache-dir pandas==2.0.3 scikit-learn==1.3.0 xgboost==2.0.0 lightgbm==4.1.0
pip install --no-cache-dir tensorflow==2.13.0 scipy==1.11.1
