### Script avec les décorateurs utilisés ###

import time

# decorateur pour mesurer le temps d'execution d'une fonction
def compute_time(function):
    def somefunction(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print(f"Temps d'exécution : {end_time - start_time:.0f} secondes")
        return result
    return somefunction