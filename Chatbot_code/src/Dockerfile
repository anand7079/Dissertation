# Use Jupyter base image
FROM jupyter/base-notebook:latest

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app
# RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Convert and execute the Jupyter notebook inside the container
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "main3_1.ipynb", "--output", "output.ipynb"]
