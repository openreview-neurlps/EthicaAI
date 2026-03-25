#!/bin/bash
echo "=== Full traceback for dmlab2d ==="
python3 -c "
import traceback
try:
    import dmlab2d
    print('OK')
except Exception:
    traceback.print_exc()
" 2>&1

echo "=== pip show dmlab2d ==="
pip3 show dmlab2d 2>&1

echo "=== site-packages location ==="
python3 -c "import site; print(site.getsitepackages())" 2>&1

echo "=== find dmlab2d files ==="
find /usr/lib/python3 /usr/local/lib/python3* /root -name '*dmlab2d*' 2>/dev/null | head -10

echo "=== DONE ==="
