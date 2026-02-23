# Add this code at the top of your script
import os
import sys

venv_path = r'C:\Users\User\PycharmProjects\PythonProject\.venv1\Scripts'
activate_this = os.path.join(venv_path, 'activate_this.py')

with open(activate_this) as f:
    exec(f.read(), {'__file__': activate_this})

print("Virtual environment activated!")
