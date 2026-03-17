#!/bin/bash
source ~/ethicaai_env/bin/activate 2>/dev/null || true
python3 -c "
try:
    import meltingpot
    print('meltingpot OK:', meltingpot.__version__)
except ImportError:
    print('NEED_INSTALL')
try:
    import dmlab2d
    print('dmlab2d OK')
except ImportError:
    print('dmlab2d MISSING')
import sys
print('Python:', sys.executable)
"
