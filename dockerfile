FROM python:3.10.10
WORKDIR /app
COPY requirements.txt .
# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Manually download and install pandas_ta
RUN wget https://github.com/twopirllc/pandas-ta/archive/refs/tags/v0.3.16b0.tar.gz && \
    tar -xzf v0.3.16b0.tar.gz && \
    cd pandas-ta-0.3.16b0 && \
    pip install --no-cache-dir . && \
    cd .. && rm -rf pandas-ta-0.3.16b0 v0.3.16b0.tar.gz
COPY . .
CMD ["python", "Ranked_ML.py"]
