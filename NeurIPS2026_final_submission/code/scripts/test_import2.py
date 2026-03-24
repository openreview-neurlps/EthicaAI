import traceback
import sys

with open('/tmp/mp_import_error.txt', 'w') as f:
    try:
        import meltingpot
        f.write("OK\n")
    except Exception as e:
        traceback.print_exc(file=f)
