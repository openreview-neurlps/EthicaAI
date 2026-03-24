import traceback
try:
    import meltingpot
    print("meltingpot OK")
except Exception as e:
    traceback.print_exc()
