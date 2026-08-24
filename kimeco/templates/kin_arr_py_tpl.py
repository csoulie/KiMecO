kin_arr_py_tpl = """
FORMATTED_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
python {filename}P${{FORMATTED_ID}}.py
"""
